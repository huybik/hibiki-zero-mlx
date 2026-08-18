# Direct VI-EN training loss

This document describes the teacher-forced objective implemented by
`finetune/common.py` for direct, simultaneous Vietnamese-speech-to-English-speech
training.

## Training streams

Each cached example contains aligned streams for:

- CTC-timed English text, including its text EOS;
- English target Mimi audio codes;
- Vietnamese source Mimi audio codes, including the source-audio EOS.

The Vietnamese audio is conditioning context. The supervised targets are the
English text and English target audio streams; there is no reconstruction loss
on the Vietnamese source audio.

Let:

- \(b\) index examples in a batch;
- \(t\) index aligned time steps;
- \(q\) index English target-audio codebooks;
- \(y_{bt}\) be a gold CTC-timed English text token;
- \(a_{bqt}\) be a gold English Mimi audio token;
- \(z^y_{bt}\) and \(z^a_{bqt}\) be the corresponding model logits.

## Teacher forcing

Teacher forcing is the way the model is conditioned during training, not an
additional loss term. The unmodified gold target streams remain in the model
input. Conceptually, a prediction at time \(t\) is conditioned on the Vietnamese
source available at that point and on the gold English target history:

\[
z_t = f_\theta\!\left(
    x^{vi}_{\operatorname{available}(t)},
    y^{en,*}_{<t},
    a^{en,*}_{<t}
\right).
\]

The asterisk denotes ground truth. At inference time, the gold English history
is unavailable, so the model conditions on its own generated English tokens
instead.

For logits \(z\) and gold class \(c\), the token-level cross-entropy is

\[
\operatorname{CE}(z,c)
= -\log \operatorname{softmax}(z)_c
= -\log\frac{e^{z_c}}{\sum_v e^{z_v}}.
\]

## English target-audio loss

Let \(M^a_{bqt}\) be one when an English target-audio token is valid and zero
when that position is masked. The audio loss is the mean cross-entropy over all
valid target-audio tokens, time steps, examples, and Mimi codebooks:

\[
N_a = \sum_{b,q,t} M^a_{bqt},
\]

\[
L_{\mathrm{audio}}
= \frac{1}{N_a}
  \sum_{b,q,t} M^a_{bqt}
  \operatorname{CE}\!\left(z^a_{bqt},a_{bqt}\right).
\]

All codebooks share this global mean. Positions introduced only to collate
different sequence lengths are excluded by the audio mask and produce no loss.

## English text masks

There are two different notions of padding:

1. **Text PAD tokens** are real tokens in the CTC-timed English stream. They
   represent frames at which no English text token is emitted.
2. **Batch padding** extends shorter tensors to the batch length. These invalid
   positions are outside `output.text_mask` and never receive a loss.

The run uses `text_pad_mode=prefix`. Let \(M^t_{bt}\) be the base valid-text
mask and let

\[
S_{bt}
= \mathbf{1}\!\left[\exists s \leq t : y_{bs} \neq \mathrm{PAD}\right]
\]

mean that a non-PAD text token has appeared by time \(t\). The content and
supervised-prefix-PAD masks are

\[
M^c_{bt}
= M^t_{bt}\,\mathbf{1}[y_{bt}\neq\mathrm{PAD}],
\]

\[
M^p_{bt}
= M^t_{bt}\,\mathbf{1}[y_{bt}=\mathrm{PAD}]\,(1-S_{bt}).
\]

Consequently:

- every valid non-PAD English token is supervised;
- the English text EOS is a non-PAD content token and is supervised;
- PAD positions before the first text token are supervised by the PAD branch;
- PAD positions after the first text token are ignored in `prefix` mode;
- invalid collation padding is always ignored.

## English text losses

The content loss is independently averaged over valid non-PAD text tokens:

\[
N_c = \sum_{b,t}M^c_{bt},
\]

\[
L_{\mathrm{content}}
= \frac{1}{N_c}
  \sum_{b,t}M^c_{bt}
  \operatorname{CE}\!\left(z^y_{bt},y_{bt}\right).
\]

The prefix-padding loss is independently averaged over supervised PAD tokens:

\[
N_p = \sum_{b,t}M^p_{bt},
\]

\[
L_{\mathrm{pad}}
= \frac{1}{N_p}
  \sum_{b,t}M^p_{bt}
  \operatorname{CE}\!\left(z^y_{bt},\mathrm{PAD}\right).
\]

The two means are combined with padding weight \(\lambda_p=0.05\):

\[
L_{\mathrm{text}}
= \frac{L_{\mathrm{content}} + \lambda_p L_{\mathrm{pad}}}
       {1+\lambda_p}
= \frac{L_{\mathrm{content}} + 0.05L_{\mathrm{pad}}}{1.05}.
\]

This gives the content component \(1/1.05\approx95.24\%\) and the PAD component
\(0.05/1.05\approx4.76\%\) of the text branch. Because each component is
averaged before they are mixed, a large number of PAD frames cannot overwhelm
the content objective.

## Total training loss

The general implemented objective is

\[
L_{\mathrm{TF}}
= \lambda_a L_{\mathrm{audio}}
+ \lambda_t L_{\mathrm{text}}.
\]

The direct recipe uses \(\lambda_a=1\) and \(\lambda_t=1\), so

\[
\boxed{
L_{\mathrm{TF}}
= L_{\mathrm{audio}}
+ \frac{L_{\mathrm{content}}+0.05L_{\mathrm{pad}}}{1.05}
}.
\]

The audio and text branches are added; the result is not divided by two. The
objective contains no ASR replay, contrastive, anti-repetition, or post-source
transformation loss.

## Step-135 example

The teacher-forced validation metrics at checkpoint step 135 were:

| Component | Loss |
| --- | ---: |
| English target audio | 2.676538 |
| English text content | 4.830242 |
| English prefix PAD | 0.030508 |

The combined text loss is

\[
L_{\mathrm{text}}
= \frac{4.830242+0.05(0.030508)}{1.05}
= 4.601683.
\]

The reported total is therefore

\[
L_{\mathrm{TF}}
= 2.676538+4.601683
= 7.278221.
\]

This metric measures next-token prediction under gold target history. It does
not by itself measure free-running translation quality; BLEU, chrF, generation
health, and audio evaluation test behavior after the model must consume its own
predictions.
