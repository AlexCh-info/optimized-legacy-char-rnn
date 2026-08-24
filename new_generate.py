import torch
import os
import asyncio

from helpers import *
from model import *

class Generation:
    def __init__(self, decoder, cuda: bool = False) -> None:
        self.decoder = decoder
        self.hidden = self.decoder.init_hidden(1)
        self.cuda = cuda

    async def warm_up(self, prime_str: str) -> None:
        """To put it simply, "warming up" the model involves adding context before the
        main generation phase. This is necessary to obtain a hidden state that already
        reflects the initial context of the subsequent text, rather than one that is merely random."""

        prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
        if self.cuda:
            self.hidden = self.hidden.cuda()
            prime_input = prime_input.cuda()

        self.inp = await asyncio.to_thread(self._warm_up_with_no_grad, len(prime_str), prime_input)


    async def step(self, temperature: float = 0.8) -> str:
        if self.cuda:
            self.inp = self.inp.cuda()
        # Heavy computations block the event loop, so we move them to a separate thread.
        output, self.hidden = await asyncio.to_thread(self._forward_with_no_grad)

        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1).item()

        predicted_char = all_characters[top_i]
        self.inp = Variable(char_tensor(predicted_char).unsqueeze(0))

        return predicted_char

    @torch.no_grad
    def _forward_with_no_grad(self):
        """Wrapper for decoder"""
        return self.decoder(self.inp, self.hidden)

    @torch.no_grad
    def _warm_up_with_no_grad(self, lenght_prime, prime_input):
        """Wrapper on warm_up method"""
        for p in range(lenght_prime - 1):
            _, self.hidden = self.decoder(prime_input[:, p], self.hidden)
        # ToDo if user will pass on argument prime input as "", function drops with error IndexError
        return prime_input[:, -1]

