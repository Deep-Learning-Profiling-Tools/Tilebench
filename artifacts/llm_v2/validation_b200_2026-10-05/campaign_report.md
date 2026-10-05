| task | model | status | rounds | valid | attempts | repairs | best ms (round) | samples | tokens (in/out/reasoning) | exact | reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cutile/histogramming/int32 | claude | complete | 10 | 10 | 10 | 0 | 0.1030 (10) | 0.1033, 0.1031, 0.1026 | 260928/286645/None | True | - |
| cutile/softmax/fp16 | claude | complete | 10 | 7 | 10 | 0 | 0.0167 (9) | 0.0168, 0.0167, 0.0166 | 252472/154841/None | False | - |
| cutile/vector_add/fp16 | claude | complete | 10 | 10 | 10 | 0 | 0.0218 (1) | 0.0218, 0.0218, 0.0217 | 231538/36783/None | True | - |
| cutile/histogramming/int32 | gpt | complete | 10 | 10 | 10 | 0 | 0.0918 (10) | 0.0923, 0.0920, 0.0911 | 158739/49967/42368 | True | - |
| cutile/softmax/fp16 | gpt | complete | 10 | 9 | 10 | 0 | 0.0171 (8) | 0.0175, 0.0170, 0.0169 | 156343/50325/44052 | True | r2:compliant, r3:compliant |
| cutile/vector_add/fp16 | gpt | complete | 10 | 10 | 10 | 0 | 0.0217 (8) | 0.0216, 0.0217, 0.0219 | 149208/29534/26185 | True | r5:compliant |
| triton/histogramming/int32 | claude | complete | 10 | 7 | 10 | 0 | 0.0751 (5) | 0.0762, 0.0746, 0.0744 | 261004/538209/None | True | r2:compliant, r3:compliant |
| triton/softmax/fp16 | claude | complete | 10 | 10 | 10 | 0 | 0.0162 (1) | 0.0167, 0.0159, 0.0161 | 274932/368662/None | True | r1:compliant, r9:compliant |
| triton/vector_add/fp16 | claude | complete | 10 | 9 | 10 | 0 | 0.0205 (7) | 0.0182, 0.0215, 0.0216 | 238080/176547/None | True | - |
| triton/histogramming/int32 | gpt | complete | 10 | 10 | 10 | 0 | 0.0699 (7) | 0.0711, 0.0694, 0.0691 | 162041/57960/48995 | True | - |
| triton/softmax/fp16 | gpt | complete | 10 | 10 | 10 | 0 | 0.0165 (10) | 0.0165, 0.0165, 0.0164 | 162267/44595/37016 | True | r1:compliant, r2:compliant |
| triton/vector_add/fp16 | gpt | complete | 10 | 10 | 10 | 0 | 0.0219 (10) | 0.0219, 0.0219, 0.0219 | 151350/26928/23251 | True | - |
