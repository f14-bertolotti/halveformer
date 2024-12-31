# Halve Transformer
Repository for the implementation of the Halve Transformer

## Idea
A self-attention layer in trasformers generates an attention matrix that is used to weight the token sequence. 
If one can halve the attention matrix then the sequence generated will also be halved.
Of course, one wants to halve the attention matrix intelligently.
Now, recall that the attention matrix is often mostly diagonal, at least, in the early layers of the transformer.
So why not leverage this fact and simply average rows 2 by 2 in the attention matrix?

## Implementation

The implementation is of the halve transformer is based on halving the attention matrix by averaging rows 2 by 2.
The full trick can be implemented with one additional line of code to the multi-head attention layer.

```python
if self.halve:
    attention_output = attention_output.view(attention_output.size(0), attention_output.size(1)//2, -1, attention_output.size(2)).mean(1)
```

Of course, you can also use $k$ instead of 2.

## Reproducibility
If you want to reproduce these experiments, it should suffice to run:
```bash
make -f makefile.mk all
```

## Results
This simple trick can be used to reduce the number of tokens in the sequence by a factor of 2 (or $k$). 
This can be beneficial in terms of memory usage of the self-attention layer.
I have a little bit of experience with this trick and it seems to work well in practice.
However, these test are run on a small dataset ([emotion](https://huggingface.co/datasets/dair-ai/emotion)) and the results are not conclusive.

![alt text](https://github.com/f14-bertolotti/halveformer/blob/main/figs/fig.png?raw=true)

It can be seen that the HalveTransformer performs on par with the Vanilla Transformer. 
However, it reduces the memory usage quite a bit. You should also expect better savings with longer sequence lengths.
On the other hand, you get slightly worse performance due to the additional operation in the self attention.


