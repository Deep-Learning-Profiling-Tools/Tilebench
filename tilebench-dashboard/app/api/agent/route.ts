import { fail } from "@/lib/http";
import { AGENT_TOOLS, runAgentTool, toolPreview } from "@/lib/agent/tools";

export const maxDuration = 120;

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const MODEL = process.env.TILEBENCH_AGENT_MODEL ?? "openai/gpt-5.6-luna";
const MAX_QUESTION = 4000;
const MAX_HISTORY_MESSAGES = 20;
const MAX_HISTORY_CHARS = 24_000;
const MAX_MESSAGE_CHARS = 8_000;
const MAX_TOOL_ROUNDS = 10;
const MAX_OUTPUT_TOKENS = 4096;

const SYSTEM_PROMPT = `You are the TileBench++ performance analysis agent.

TileBench++ benchmarks the same 45 GPU kernels written in Triton, cuTile and TileLang against PyTorch on several hardware platforms: NVIDIA B200, NVIDIA GH200 and AMD Instinct MI300X (Triton only on MI300X). Treat the backends even-handedly; none of them is the subject by default.

You answer only from evidence the tools return: per-platform benchmark results, each backend's implementation source, and the exported profiling reports (Nsight Compute on B200 and GH200, rocprof-compute on MI300X), one per platform, operator, backend and dtype.

Reading the benchmark numbers:
- Every board figure is a speedup over PyTorch on the same hardware and inputs: PyTorch time divided by backend time, as a geometric mean over the kernel's cases. 2.0 means twice as fast as PyTorch; 0.5 means half as fast. State this when you quote a figure, and write figures with a x suffix.
- default and autotune are separate measurements. Never average or blend them; always say which one a number is.
- Speedups are relative to PyTorch on that platform, so they do not compare absolute latency across platforms.

Explaining performance:
- A claim about why a kernel is fast or slow must rest on profiler measurements, not on reading the code alone. Call get_profile_summary for the reports involved, then query_profile_metrics or compare_profile_metrics for the specific counters, and quote the values with their metric names.
- Each profiling report is one representative input size at the autotuned configuration, while board speedups are means over many sizes. Say so when you connect the two.
- Separate what the measurements show from what you infer. If the evidence does not single out a cause, say that the cause is not established and list what the data rules in or out.
- You cannot run code, re-profile, or see compiler internals for any backend. Do not explain behaviour by describing what a compiler "must have" generated; point to the measured instruction hotspots, which are assembly-level on B200 and GH200 and carry Python source lines only on MI300X, and say when a report has none.
- Metric names differ between Nsight Compute and rocprof-compute and between GPU generations. Compare backends within one platform first; treat cross-platform metric comparisons with care.

Rules:
- Use tools before making any factual claim about operators, results, source or profiler data.
- For any question about a platform, or comparing platforms, call get_platform_results. If a platform is selected, answer for that platform unless asked otherwise.
- If a selected operator is provided, treat it as the default subject and call get_operator for it before analyzing it.
- To explain a gap between backends, read both implementations with read_source and compare their profiles, rather than reasoning from one side.
- Read large files in chunks with offset/next_offset when the first chunk is insufficient.
- If a report, metric or source file is missing, say exactly what is missing instead of guessing.
- Cite the concrete platform/op/backend/dtype, metric names and file names you used.`;

type JsonObject = Record<string, unknown>;

type ToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments?: string;
  };
};

type ChatMessage =
  | { role: "system" | "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[] }
  | { role: "tool"; tool_call_id: string; name: string; content: string };

type OpenRouterMessage = {
  role?: string;
  content?: unknown;
  tool_calls?: ToolCall[];
};

type ClientMessage = { role: "user" | "assistant"; content: string };

const OPENROUTER_TOOLS = AGENT_TOOLS.map((tool) => ({
  type: "function" as const,
  function: {
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema,
  },
}));

function openRouterHeaders(key: string): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${key}`,
    "Content-Type": "application/json",
    "X-OpenRouter-Title": process.env.OPENROUTER_APP_TITLE ?? "TileBench++ Dashboard",
  };
  const referer = process.env.OPENROUTER_SITE_URL ?? process.env.VERCEL_PROJECT_PRODUCTION_URL;
  if (referer) headers["HTTP-Referer"] = referer.startsWith("http") ? referer : `https://${referer}`;
  return headers;
}

function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) return String((part as JsonObject).text ?? "");
        return JSON.stringify(part);
      })
      .filter(Boolean)
      .join("\n");
  }
  return JSON.stringify(content);
}

function parseToolArgs(raw: string | undefined): unknown {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { __parse_error: "tool arguments were not valid JSON", raw };
  }
}

function trimClientMessages(messages: ClientMessage[]): ClientMessage[] {
  const valid = messages
    .filter((m) => (m.role === "user" || m.role === "assistant") && m.content.trim())
    .slice(-MAX_HISTORY_MESSAGES);
  const kept: ClientMessage[] = [];
  let chars = 0;
  for (let i = valid.length - 1; i >= 0; i--) {
    const msg = valid[i];
    const remaining = MAX_HISTORY_CHARS - chars;
    if (remaining <= 0) break;
    const content = msg.content.length > remaining ? msg.content.slice(-remaining) : msg.content;
    kept.push({ role: msg.role, content });
    chars += content.length;
  }
  return kept.reverse();
}

function parseClientMessages(raw: unknown): { messages: ClientMessage[]; error?: string } {
  if (raw == null) return { messages: [] };
  if (!Array.isArray(raw)) return { messages: [], error: "messages must be an array" };
  const messages: ClientMessage[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return { messages: [], error: "each message must be an object" };
    }
    const row = item as Record<string, unknown>;
    if (row.role !== "user" && row.role !== "assistant") {
      return { messages: [], error: "message role must be user or assistant" };
    }
    if (typeof row.content !== "string") {
      return { messages: [], error: "message content must be a string" };
    }
    const content = row.content.trim();
    if (!content) continue;
    if (content.length > MAX_MESSAGE_CHARS) {
      return { messages: [], error: "one message is too long" };
    }
    messages.push({ role: row.role, content });
  }
  return { messages: trimClientMessages(messages) };
}

async function complete(
  key: string,
  messages: ChatMessage[],
  toolChoice: "auto" | "required" | "none",
): Promise<OpenRouterMessage> {
  const body: JsonObject = {
    model: MODEL,
    messages,
    max_tokens: MAX_OUTPUT_TOKENS,
    temperature: 0.1,
  };
  if (toolChoice !== "none") {
    body.tools = OPENROUTER_TOOLS;
    body.tool_choice = toolChoice;
  }

  const res = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: openRouterHeaders(key),
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`OpenRouter request failed (${res.status}): ${text.slice(0, 500)}`);
  }

  const json = (await res.json()) as {
    choices?: Array<{ message?: OpenRouterMessage; finish_reason?: string }>;
    error?: { message?: string };
  };
  if (json.error?.message) throw new Error(json.error.message);
  const msg = json.choices?.[0]?.message;
  if (!msg) throw new Error("OpenRouter returned no message");
  return msg;
}

/**
 * POST /api/agent  { question?, messages?: Array<{role, content}>, contextOp?, contextPlatform?, apiKey? }
 * Streams Server-Sent Events: `meta`, zero or more `tool`, `delta`, then `done`.
 */
export async function POST(req: Request) {
  let payload: {
    question?: unknown;
    messages?: unknown;
    contextOp?: unknown;
    contextPlatform?: unknown;
    apiKey?: unknown;
  };
  try {
    payload = await req.json();
  } catch {
    return fail(400, "body must be JSON");
  }

  const suppliedKey = typeof payload.apiKey === "string" ? payload.apiKey.trim() : "";
  const key = suppliedKey || process.env.OPENROUTER_API_KEY;
  if (!key) {
    return fail(401, "OpenRouter API key is required", {
      hint: "paste an OpenRouter API key in the drawer chat tab or set OPENROUTER_API_KEY in Vercel",
    });
  }

  const contextOp = typeof payload.contextOp === "string" ? payload.contextOp.trim() : "";
  if (contextOp && !/^[a-z0-9_]+$/.test(contextOp)) return fail(400, "bad context operator");
  const contextPlatform =
    typeof payload.contextPlatform === "string" ? payload.contextPlatform.trim() : "";
  if (contextPlatform && !/^[A-Za-z0-9_]{1,24}$/.test(contextPlatform)) {
    return fail(400, "bad context platform");
  }
  const parsedMessages = parseClientMessages(payload.messages);
  if (parsedMessages.error) return fail(400, parsedMessages.error);

  const question = typeof payload.question === "string" ? payload.question.trim() : "";
  if (question.length > MAX_QUESTION) return fail(413, "question too long");
  let conversation = parsedMessages.messages;
  if (conversation.length === 0) {
    if (!question) return fail(400, "question or messages are required");
    conversation = [{ role: "user", content: question }];
  } else if (conversation[conversation.length - 1]?.role !== "user") {
    if (!question) return fail(400, "last message must be user");
    conversation = trimClientMessages([...conversation, { role: "user", content: question }]);
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (event: string, data: unknown) =>
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));

      const messages: ChatMessage[] = [
        { role: "system", content: SYSTEM_PROMPT },
        ...(contextPlatform
          ? [{
              role: "system" as const,
              content: `Selected platform: ${contextPlatform}. The user is looking at this platform's results.`,
            }]
          : []),
        ...(contextOp
          ? [{
              role: "system" as const,
              content: `Selected operator: ${contextOp}. Treat it as the default subject for ambiguous follow-up questions.`,
            }]
          : []),
        ...conversation.map((message): ChatMessage => ({ role: message.role, content: message.content })),
      ];

      try {
        send("meta", {
          provider: "openrouter",
          model: MODEL,
          context_op: contextOp || null,
          history_messages: conversation.length,
          tools: AGENT_TOOLS.map((t) => t.name),
        });

        for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
          const msg = await complete(key, messages, round === 0 ? "required" : "auto");
          const toolCalls = msg.tool_calls ?? [];
          const content = contentText(msg.content);
          messages.push({
            role: "assistant",
            content: content || null,
            ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
          });

          if (!toolCalls.length) {
            if (content) send("delta", { text: content });
            send("done", { ok: true, tool_rounds: round });
            return;
          }

          for (const call of toolCalls) {
            const input = parseToolArgs(call.function.arguments);
            send("tool", { status: "start", name: call.function.name, id: call.id, input });
            const result = await runAgentTool(call.function.name, input);
            send("tool", {
              status: "done",
              name: call.function.name,
              id: call.id,
              ok: result.ok !== false,
              preview: toolPreview(result),
            });
            messages.push({
              role: "tool",
              tool_call_id: call.id,
              name: call.function.name,
              content: JSON.stringify(result),
            });
          }
        }

        const finalMsg = await complete(key, messages, "none");
        const finalText = contentText(finalMsg.content) || "Tool round limit reached before a final answer.";
        send("delta", { text: finalText });
        send("done", { ok: true, tool_rounds: MAX_TOOL_ROUNDS, reached_tool_round_limit: true });
      } catch (err) {
        send("error", { message: err instanceof Error ? err.message : "agent failed" });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
