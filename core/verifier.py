import torch

def verify(output, reference, atol=1e-3, rtol=1e-3):
    try:
        torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
        return True, ""
    except Exception as e:
        return False, str(e)
