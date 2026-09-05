# References and research context

Citations below were verified against Crossref / arXiv rather than written from
memory. Where a work has both a conference and a journal version, both are given.

---

## Directly related work

### 1. Doersch, Singh, Gupta, Sivic & Efros (2012) — *What makes Paris look like Paris?*

> Doersch, C., Singh, S., Gupta, A., Sivic, J., & Efros, A. A. (2012).
> What makes Paris look like Paris? *ACM Transactions on Graphics*, 31(4),
> Article 101. DOI: [10.1145/2185520.2185597](https://doi.org/10.1145/2185520.2185597)
> — reprinted as *Communications of the ACM*, 58(12), 2015.
> DOI: [10.1145/2830541](https://doi.org/10.1145/2830541)

The founding paper for this problem. It mines Google Street View with a
discriminative clustering method to find image *patches* that are both frequent
in a city and rare elsewhere, recovering interpretable architectural elements —
Parisian cast-iron balcony railings, particular window and doorway forms, street
signage. Supervision is weak: only the geotag.

**How this project relates.** Same question, different mechanism and different
emphasis. Doersch et al. discover discriminative *patches* with an explicitly
mid-level, part-based method; we train a whole-image encoder end-to-end and then
interrogate its embedding, so our "elements" are model-discovered modes rather
than localised parts. The larger difference is scope: their work establishes
*that* geographically discriminative elements exist, whereas roughly half of our
effort goes to asking whether what our model found is architecture at all, or the
camera. We do not claim to improve on their method.

### 2. Guo, Jang, Duarte, Kang & Ratti (2025) — Urban visual uniqueness, landmark-free

> Guo, S., Jang, K. M., Duarte, F., Kang, Y., & Ratti, C. (2025).
> Urban visual uniqueness: A landmark-free framework to quantify city's identity
> and distinctiveness from everyday scenes.
> *Computers, Environment and Urban Systems*, 122, 102351.
> DOI: [10.1016/j.compenvurbsys.2025.102351](https://doi.org/10.1016/j.compenvurbsys.2025.102351)

Explicitly rejects landmarks in favour of everyday scenes. Extracts features from
a **pretrained** network, clusters them unsupervised into visual character types,
and quantifies each city's *identity* (which types it contains) and
*distinctiveness* (how far its mixture deviates from other cities).

**How this project relates.** The closest match in motivation — the
"ordinary streets, not landmarks" premise is shared, and their identity /
distinctiveness split is close to our characteristic / atypical framing. Two
differences. First, they use frozen pretrained features; **we train the encoder**,
which lets us ask what a city-supervised objective learns that a generic one does
not — and our corrected retrieval evaluation found the trained encoder *worse*
than frozen ImageNet for cross-city physical similarity. Second, we run an
explicit shortcut audit; their clustering, like ours, would be susceptible to
capture-campaign structure, which is a general hazard for this family of methods
rather than a criticism of that paper.

### 3. Zhang, Yang, He, Li & Yang (2026) — City identity recognition and representation bias

> Zhang, X., Yang, F., He, Z., Li, W., & Yang, M. (2026).
> City identity recognition: how representation bias influences model
> predictability and replicability?
> *Computers, Environment and Urban Systems*, 123, 102370.
> DOI: [10.1016/j.compenvurbsys.2025.102370](https://doi.org/10.1016/j.compenvurbsys.2025.102370)
> (published online 2025; issue dated January 2026)

Asks whether models that recognise city identity from crowd-sourced imagery are
actually measuring the city, given that the imagery is unevenly and
non-randomly collected. Frames the problem as *weak replicability*: a model
degrades or misleads when moved to data collected differently from its training
data.

**How this project relates.** The most direct methodological parallel to our
second half. Our shortcut audit is an independent, concrete instance of exactly
their concern: capture metadata alone predicts city at **0.9249 on our data,
exceeding the pixel model's 0.9063**. Our cross-campaign holdout is our attempt at
their replicability test — and it is where we found that our own first version of
that test was contaminated. We regard their framing as the correct one for
interpreting any city-identity accuracy number, including ours.

### 4. Alpherts, Ghebreab & van Noord (2025) — Artifacts of idiosyncrasy in global street view data

> Alpherts, T., Ghebreab, S., & van Noord, N. (2025).
> Artifacts of Idiosyncracy in Global Street View Data.
> arXiv:[2505.11046](https://arxiv.org/abs/2505.11046)

Examines street view data across 28 cities and finds systematic collection
biases that survive dense coverage — biases arising from a city's own layout and
from how collection is carried out, not merely from missing areas.

**How this project relates.** Supplies the mechanism behind our audit's
findings. Our per-city campaign concentration (Casablanca 92.7% one month;
fisheye share 0.000–0.837 across cities) is a concrete instance of collection
idiosyncrasy, and it is why we treat unflagged visual modes as *supporting* a
signature interpretation rather than *establishing* one.

---

## Method and data sources

> Hou, Y., Quintana, M., Khomiakov, M., Yap, W., Ouyang, J., Ito, K., Wang, Z.,
> Zhao, T., et al. (2024). Global Streetscapes — A comprehensive dataset of 10
> million street-level images across 688 cities for urban science and analytics.
> *ISPRS Journal of Photogrammetry and Remote Sensing*, 215, 216–238.
> DOI: [10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)

The dataset. Also the source of the physical attributes used for interpretation
(`green_view_index`, `Building`/`Total`, `type_highway`) and of the perception
scores this project deliberately does not use.

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image
> Recognition. *CVPR*. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)

ResNet50, the backbone, via `timm`.

> Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-
> Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.
> arXiv:[2105.04906](https://arxiv.org/abs/2105.04906)

The self-supervised objective tested as an alternative pretext task. Implemented
from the paper's three-term formulation; our 100-epoch run did not significantly
beat frozen ImageNet on the retrieval metric, and we report that.

> McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
> Approximation and Projection. arXiv:[1802.03426](https://arxiv.org/abs/1802.03426)
>
> Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep
> Networks via Gradient-based Localization. *ICCV*.
> DOI: [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)

Visualisation and attribution tooling.

---

## Positioning

This project sits at the intersection of three lines of work:

**Geographically discriminative visual-element discovery** (Doersch et al.) —
the idea that places have recoverable, interpretable visual signatures.

**Landmark-free urban visual identity and distinctiveness** (Guo et al.) — the
insistence that ordinary scenes, not monuments, carry a city's character.

**Representation bias and shortcut auditing** (Zhang et al.; Alpherts et al.) —
the discipline of asking whether a model that recognises a place is measuring
the place or the data collection.

### What this project does and does not claim

**We do not claim methodological novelty.** The encoder is a standard ResNet50
with a standard classification head; the signature analysis is centroid margins
and k-means; VICReg is implemented from its paper. None of these is new.

What the project offers is a **combination that is uncommon in practice**: a
city-discriminative encoder trained from scratch, evaluated for cross-city
transfer, *and* audited against capture confounds on the same data, with the
negative results and self-corrections kept in the record — a failed retrieval
objective, a self-inflicted evaluation defect, a self-supervised model that did
not beat an untrained baseline, and a first campaign test that had to be
discarded as contaminated.

The strongest defensible statement is the narrow one:

> The dominant temporal/capture metadata confound does not explain the encoder's
> performance on the clean cross-campaign subset.

Not that the representation is confound-free, and not that the visual modes it
exposes are intrinsic properties of cities.

### Claim ladder

| Level | Claim |
|---|---|
| **Supported** | The trained representation contains strong city-discriminative visual information that transfers to genuinely unseen later-period imagery. |
| **Suggestive** | Some ordinary street patterns are strongly associated with city identity in the learned representation. |
| **Not established** | That the discovered patterns are causal, exhaustive, uniquely intrinsic to a city, or free of every possible collection artefact. |
