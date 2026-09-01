import { NextResponse } from "next/server";

export function ok<T>(data: T, init?: ResponseInit) {
  return NextResponse.json(data, init);
}
export function fail(status: number, error: string, extra?: Record<string, unknown>) {
  return NextResponse.json({ error, ...extra }, { status });
}
/** Operator/file segments are used to build paths — keep them boring. */
export const SAFE = /^[a-z0-9_]+$/;
export const SAFE_FILE = /^[a-zA-Z0-9_.]+$/;
