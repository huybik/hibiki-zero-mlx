/* Exact CTC Viterbi forced alignment of one clip's log-probabilities to a known token sequence
 * (timing.CTCWordAligner.align, which compiles and loads this file).
 *
 * A path skips the blank between two different tokens. On ties a state keeps the first candidate
 * of [stay, advance, skip]: stay (0) beats advance (1), which beats skip (2).
 * Ending state: states - 1 if score[states - 1] >= score[states - 2] else states - 2.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Returns 0 on success, -1 on allocation failure, -2 if any target token got 0 frames. */
int ctc_viterbi_align(
    const float *log_probs,   /* [T, V] row-major */
    int T,
    int V,
    const int32_t *targets,   /* [L] */
    int L,
    int blank_id,
    int32_t *out_token_start, /* [L] min frame per target */
    int32_t *out_token_end,   /* [L] max frame per target */
    double *out_score         /* mean posterior over aligned non-blank frames */
) {
    if (L <= 0 || T <= 0 || V <= 0) {
        return -2;
    }
    const int S = 2 * L + 1;
    int32_t *symbols = (int32_t *)malloc((size_t)S * sizeof(int32_t));
    uint8_t *can_skip = (uint8_t *)calloc((size_t)S, sizeof(uint8_t));
    float *score_prev = (float *)malloc((size_t)S * sizeof(float));
    float *score_curr = (float *)malloc((size_t)S * sizeof(float));
    int8_t *back = (int8_t *)calloc((size_t)T * (size_t)S, sizeof(int8_t));
    if (!symbols || !can_skip || !score_prev || !score_curr || !back) {
        free(symbols);
        free(can_skip);
        free(score_prev);
        free(score_curr);
        free(back);
        return -1;
    }

    for (int s = 0; s < S; s++) {
        symbols[s] = (s & 1) ? targets[s >> 1] : blank_id;
        score_prev[s] = -INFINITY;
    }
    for (int s = 3; s < S; s += 2) {
        if (symbols[s] != symbols[s - 2]) {
            can_skip[s] = 1;
        }
    }

    score_prev[0] = log_probs[symbols[0]];
    score_prev[1] = log_probs[symbols[1]];

    for (int t = 1; t < T; t++) {
        const float *row = log_probs + (size_t)t * (size_t)V;
        int8_t *b_row = back + (size_t)t * (size_t)S;

        /* s = 0: only stay */
        score_curr[0] = score_prev[0] + row[symbols[0]];
        b_row[0] = 0;

        /* s = 1: stay or advance */
        if (S > 1) {
            float c0 = score_prev[1];
            float c1 = score_prev[0];
            if (c1 > c0) {
                score_curr[1] = c1 + row[symbols[1]];
                b_row[1] = 1;
            } else {
                score_curr[1] = c0 + row[symbols[1]];
                b_row[1] = 0;
            }
        }

        for (int s = 2; s < S; s++) {
            float best = score_prev[s];
            int8_t choice = 0;
            float adv = score_prev[s - 1];
            if (adv > best) {
                best = adv;
                choice = 1;
            }
            if (can_skip[s]) {
                float skp = score_prev[s - 2];
                if (skp > best) {
                    best = skp;
                    choice = 2;
                }
            }
            score_curr[s] = best + row[symbols[s]];
            b_row[s] = choice;
        }

        float *tmp = score_prev;
        score_prev = score_curr;
        score_curr = tmp;
    }

    int state = (score_prev[S - 1] >= score_prev[S - 2]) ? (S - 1) : (S - 2);

    for (int i = 0; i < L; i++) {
        out_token_start[i] = T;
        out_token_end[i] = -1;
    }

    double prob_sum = 0.0;
    int64_t prob_count = 0;

    for (int t = T - 1; t >= 0; t--) {
        if (state & 1) {
            int idx = state >> 1;
            if (t < out_token_start[idx]) {
                out_token_start[idx] = t;
            }
            if (t > out_token_end[idx]) {
                out_token_end[idx] = t;
            }
            float lp = log_probs[(size_t)t * (size_t)V + (size_t)targets[idx]];
            prob_sum += (double)expf(lp);
            prob_count++;
        }
        state -= (int)back[(size_t)t * (size_t)S + (size_t)state];
    }

    int rc = 0;
    for (int i = 0; i < L; i++) {
        if (out_token_end[i] < 0) {
            rc = -2;
            break;
        }
    }
    if (rc == 0 && prob_count > 0) {
        *out_score = prob_sum / (double)prob_count;
    } else {
        *out_score = 0.0;
        rc = -2;
    }

    free(symbols);
    free(can_skip);
    free(score_prev);
    free(score_curr);
    free(back);
    return rc;
}
