import unittest

import torch

import prepare
import train


class TrainCompatibilityTest(unittest.TestCase):
    def test_norm_supports_torch_without_functional_rms_norm(self):
        x = torch.randn(2, 3, 5, dtype=torch.float32)

        y = train.norm(x)

        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + torch.finfo(x.dtype).eps)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-5))

    def test_cpu_eval_budget_is_smaller_than_default(self):
        eval_tokens = train.resolve_eval_tokens(torch.device("cpu"))

        self.assertLess(eval_tokens, prepare.EVAL_TOKENS)
        self.assertEqual(eval_tokens, train.CPU_EVAL_TOKENS)


if __name__ == "__main__":
    unittest.main()
