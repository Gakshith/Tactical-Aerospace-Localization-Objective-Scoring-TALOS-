# AIR-MOT data card

Jilin-1 satellite videos with aircraft and ships, boxes per frame, track IDs.
Paper: He, Sun, Yan, Li, Fu, "Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling", IEEE TGRS 60, 2022, [doi:10.1109/TGRS.2022.3152250](https://doi.org/10.1109/TGRS.2022.3152250). Repo: [HeQibin/TGraM](https://github.com/HeQibin/TGraM). The repo has a README only: no code, no loader, no split files, no license.

**[M]** means we measured it on Polaris on 2026-11-11 with `~/talos/data/scripts/inspect_airmot.py`. Everything else is quoted from a source, and the source is named.

## Which release this is

| item | value |
|---|---|
| Release in the README | 70 videos, Drive folder `1zfvhKOGmvZVWUgbE8l3LL9uude0bBg-O`. **Dead**: HTTP 404 on 2026-11-11 (repo issues [#6](https://github.com/HeQibin/TGraM/issues/6) and [#7](https://github.com/HeQibin/TGraM/issues/7) also report it). |
| Release we use | **AIR-MOT-100**, one zip, Drive file `1xPyTejb7vtZgJzWTFJHBMEW_B7rybid8`, posted in [issue #9](https://github.com/HeQibin/TGraM/issues/9) (2024-11-17) by GitHub user `qibin-he`. The account calls it a reorganised AIR-MOT with 100 videos. It is not the repo-owner account (`HeQibin`), so its authorship is likely but not verified. |
| Zip | `AIR-MOT-100.zip`, 2,179,478,075 B, sha256 `2fcea088262a3b2a05bc7302501f7a7f471a5e0c96927c937de34a6f78a55471`, `unzip -t` clean **[M]** |
| Extracted | 2,296,770,824 B (2.14 GiB), 22,668 jpg + 100 gt.txt + 1 stray `63/img/Thumbs.db` **[M]** |
| License / terms | **Not stated** anywhere: no LICENSE file, and the license questions in [#11](https://github.com/HeQibin/TGraM/issues/11) and [#12](https://github.com/HeQibin/TGraM/issues/12) (2026-05) have no answer. Cite the TGRS paper. |

## Layout and format [M]

```
AIR-MOT-100/<vid>/img/000001.jpg ...   vid = 1..100
AIR-MOT-100/<vid>/gt/gt.txt
```
- Images are JPEG, 3-channel, **1920x1080 in all 22,668 frames**. Names are 6-digit, 1-based and contiguous, matching the gt frame index. Exception: videos 95-100 add a per-video suffix, `000001_<k>.jpg` with k = 0,1,3,4,7,8. Watch out: Python's `int("000001_0")` returns 10, so split on `_` before parsing.
- gt.txt is MOT-style CSV with CRLF line endings and 9 integer fields. There is no header and no seqinfo.ini.

Five raw lines (video 50):
```
1,1,24,81,24,30,1,1,1
1,1,1645,692,17,30,1,2,1
2,1,19,88,24,30,1,1,1
2,1,1644,691,17,30,1,2,1
3,1,13,95,24,30,1,1,1
```
| col | meaning | evidence |
|---|---|---|
| 1 | frame, 1-based, equals the image number | min 1; never above the image count |
| 2 | track id, **unique only within a class**. Track key = (video, class, id) | videos 50 and 52 use id 1 for both an aircraft and a ship |
| 3,4 | x, y of the box's **top-left** corner, px | crops drawn this way line up with the object; 8,801 boxes have x=0 or y=0, so boxes are clipped at the image edge |
| 5,6 | w, h, px | all > 0; no box goes past the image |
| 7 | always 1 (MOT "consider" flag) | value set {1} |
| 8 | class: **1 = aircraft, 2 = ship** | checked by eye on crops (v3 and v50 class 1 = aircraft; v50 and v61 class 2 = ship with wake). Not stated in the README |
| 9 | always 1 (visibility) | value set {1} |

Whether pixel coordinates are 0- or 1-based is not stated.

## Totals [M]

| | value |
|---|---|
| videos / frames | 100 / 22,668 (frames per video: 326 ×45, 70 ×23, 143 ×11, 72 ×8, 327 ×7, 325 ×6) |
| videos with annotations | **69** (19,940 frames). **31 have an empty gt.txt**: 1, 2, 9, 10, 34, 69-94 |
| videos with aircraft / ships / both | 32 / 39 / 2 (50 and 52); 30 aircraft-only, 37 ship-only |
| gt rows | 317,569 (aircraft 52,504; ship 265,065) |
| tracks | 1,209 (aircraft **216**, ship 993) |

### Box size, px [M]

| class | boxes | w min / med / p90 / max | h min / med / p90 / max |
|---|---|---|---|
| aircraft | 52,504 | 7 / 34 / 58 / 77 | 6 / 33 / 59 / 74 |
| ship | 265,065 | 3 / 10 / 30 / 171 | 3 / 10 / 29 / 222 |

## Motion [M]

Centre displacement is measured in image coordinates, so it includes any leftover registration drift. A typical parked aircraft drifts a median of 5.1 px over its life. Net displacement = first centre to last centre. Max-from-start gave the same counts. Mean speed = path length / frame span.

| class | tracks | net disp > 20 px | > 100 px | net disp med / p90 / max | mean speed px/frame med / p90 / max |
|---|---|---|---|---|---|
| aircraft | 216 | **26** | **14** | 5.1 / 39.7 / 1530 | 0.023 / 0.36 / 27.1 |
| ship | 993 | 177 | 33 | 10.4 / 22.5 / 400 | 0.041 / 0.15 / 1.63 |

Every aircraft track that moves more than 20 px (video, id, frames, boxes, net px, mean px/frame, median w×h):

| vid | id | frames | boxes | net disp | speed | w×h |
|---|---|---|---|---|---|---|
| 57 | 1 | 140-326 | 187 | 1530.2 | 8.25 | 14×16 |
| 49 | 1 | 8-141 | 134 | 1194.7 | 9.02 | 24×30 |
| 35 | 1 | 283-326 | 44 | 1166.2 | 27.14 | 62×61 |
| 47 | 9 | 119-198 | 80 | 566.5 | 7.30 | 37×39 |
| 46 | 23 | 272-327 | 56 | 344.6 | 6.36 | 37×39 |
| 47 | 1 | 1-327 | 327 | 289.5 | 0.92 | 42×47 |
| 31 | 8 | 1-263 | 263 | 260.5 | 1.14 | 36×45 |
| 30 | 5 | 1-326 | 326 | 228.4 | 0.86 | 33×41 |
| 45 | 2 | 1-327 | 327 | 207.0 | 0.66 | 35×43 |
| 4 | 1 | 1-143 | 143 | 203.5 | 1.52 | 32×33 |
| 30 | 6 | 1-243 | 243 | 159.6 | 0.80 | 33×35 |
| 48 | 1 | 142-159 | 18 | 154.2 | 9.14 | 25×29 |
| 47 | 7 | 1-327 | 327 | 119.9 | 0.46 | 46×37 |
| 43 | 12 | 1-327 | 327 | 105.1 | 0.33 | 34×41 |
| 6 | 1 | 34-143 | 110 | 99.6 | 1.01 | 27×33 |
| 43 | 20 | 45-327 | 283 | 85.4 | 0.32 | 60×65 |
| 8 | 10 | 1-143 | 143 | 64.0 | 0.46 | 39×33 |
| 45 | 4 | 1-327 | 327 | 59.2 | 0.22 | 39×35 |
| 44 | 2 | 1-327 | 327 | 50.4 | 0.17 | 36×33 |
| 50 | 1 | 1-7 | 7 | 46.2 | 7.96 | 24×30 |
| 3 | 1 | 1-143 | 143 | 43.9 | 0.40 | 31×28 |
| 47 | 8 | 1-327 | 327 | 43.0 | 0.14 | 36×39 |
| 7 | 15 | 1-143 | 143 | 36.4 | 0.33 | 32×29 |
| 52 | 1 | 1-5 | 5 | 30.7 | 7.96 | 31×30 |
| 30 | 7 | 296-326 | 31 | 29.0 | 1.06 | 29×29 |
| 8 | 6 | 1-33 | 33 | 22.4 | 0.80 | 27×31 |

The tracks fall into two speed groups. Eight short tracks (5-187 boxes, ≥ 6 px/frame: v57, 49, 35, 47#9, 46#23, 48, 50, 52) cross the frame quickly, consistent with airborne aircraft. The other movers run at about 0.1-1.5 px/frame, consistent with taxiing. The 14 tracks over 100 px come from 11 videos (4, 30, 31, 35, 43, 45, 46, 47, 48, 49, 57). The other 190 aircraft tracks are effectively parked.

## Per-video table [M]

`0-box fr` = frames with no box. `band fr` = frames whose bottom ≥ 8 rows are pure black.

| vid | frames | gt | air trk | ship trk | boxes | boxes/fr mean | max | 0-box fr | air >20px | ship >20px | band fr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 143 | empty | 0 | 0 | 0 | 0 | 0 | 143 | 0 | 0 | 143 |
| 2 | 143 | empty | 0 | 0 | 0 | 0 | 0 | 143 | 0 | 0 | 143 |
| 3 | 143 | yes | 1 | 0 | 143 | 1.0 | 1 | 0 | 1 | 0 | 0 |
| 4 | 143 | yes | 1 | 0 | 143 | 1.0 | 1 | 0 | 1 | 0 | 0 |
| 5 | 143 | yes | 6 | 0 | 858 | 6.0 | 6 | 0 | 0 | 0 | 0 |
| 6 | 143 | yes | 1 | 0 | 110 | 0.8 | 1 | 33 | 1 | 0 | 0 |
| 7 | 143 | yes | 17 | 0 | 2431 | 17.0 | 17 | 0 | 1 | 0 | 0 |
| 8 | 143 | yes | 14 | 0 | 1892 | 13.2 | 14 | 0 | 2 | 0 | 0 |
| 9 | 143 | empty | 0 | 0 | 0 | 0 | 0 | 143 | 0 | 0 | 0 |
| 10 | 143 | empty | 0 | 0 | 0 | 0 | 0 | 143 | 0 | 0 | 0 |
| 11 | 143 | yes | 5 | 0 | 715 | 5.0 | 5 | 0 | 0 | 0 | 0 |
| 12 | 325 | yes | 0 | 129 | 41155 | 126.6 | 129 | 0 | 0 | 63 | 0 |
| 13 | 325 | yes | 0 | 9 | 2579 | 7.9 | 9 | 0 | 0 | 6 | 0 |
| 14 | 325 | yes | 0 | 4 | 851 | 2.6 | 4 | 0 | 0 | 2 | 0 |
| 15 | 325 | yes | 0 | 13 | 3724 | 11.5 | 13 | 23 | 0 | 12 | 0 |
| 16 | 325 | yes | 0 | 27 | 7742 | 23.8 | 27 | 0 | 0 | 21 | 0 |
| 17 | 325 | yes | 0 | 1 | 325 | 1.0 | 1 | 0 | 0 | 1 | 0 |
| 18 | 326 | yes | 0 | 14 | 4564 | 14.0 | 14 | 0 | 0 | 1 | 326 |
| 19 | 326 | yes | 0 | 16 | 3033 | 9.3 | 16 | 0 | 0 | 0 | 326 |
| 20 | 326 | yes | 0 | 3 | 902 | 2.8 | 3 | 0 | 0 | 2 | 326 |
| 21 | 326 | yes | 0 | 9 | 2269 | 7.0 | 9 | 0 | 0 | 1 | 0 |
| 22 | 326 | yes | 0 | 11 | 3470 | 10.6 | 11 | 0 | 0 | 3 | 0 |
| 23 | 326 | yes | 0 | 22 | 7088 | 21.7 | 22 | 0 | 0 | 1 | 0 |
| 24 | 326 | yes | 0 | 10 | 3002 | 9.2 | 10 | 0 | 0 | 5 | 0 |
| 25 | 326 | yes | 0 | 6 | 1748 | 5.4 | 6 | 0 | 0 | 1 | 0 |
| 26 | 326 | yes | 0 | 4 | 921 | 2.8 | 4 | 0 | 0 | 4 | 0 |
| 27 | 326 | yes | 0 | 1 | 250 | 0.8 | 1 | 76 | 0 | 1 | 0 |
| 28 | 326 | yes | 0 | 13 | 4238 | 13.0 | 13 | 0 | 0 | 0 | 0 |
| 29 | 326 | yes | 3 | 0 | 978 | 3.0 | 3 | 0 | 0 | 0 | 326 |
| 30 | 326 | yes | 7 | 0 | 1091 | 3.3 | 6 | 0 | 3 | 0 | 326 |
| 31 | 326 | yes | 12 | 0 | 2791 | 8.6 | 11 | 0 | 1 | 0 | 326 |
| 32 | 326 | yes | 6 | 0 | 1956 | 6.0 | 6 | 0 | 0 | 0 | 0 |
| 33 | 326 | yes | 0 | 1 | 326 | 1.0 | 1 | 0 | 0 | 1 | 0 |
| 34 | 326 | empty | 0 | 0 | 0 | 0 | 0 | 326 | 0 | 0 | 0 |
| 35 | 326 | yes | 1 | 0 | 44 | 0.1 | 1 | 282 | 1 | 0 | 0 |
| 36 | 326 | yes | 1 | 0 | 326 | 1.0 | 1 | 0 | 0 | 0 | 0 |
| 37 | 326 | yes | 0 | 2 | 481 | 1.5 | 2 | 0 | 0 | 2 | 0 |
| 38 | 326 | yes | 0 | 1 | 171 | 0.5 | 1 | 155 | 0 | 1 | 0 |
| 39 | 326 | yes | 1 | 0 | 326 | 1.0 | 1 | 0 | 0 | 0 | 0 |
| 40 | 326 | yes | 0 | 3 | 978 | 3.0 | 3 | 0 | 0 | 3 | 0 |
| 41 | 327 | yes | 3 | 0 | 981 | 3.0 | 3 | 0 | 0 | 0 | 327 |
| 42 | 327 | yes | 2 | 0 | 650 | 2.0 | 2 | 0 | 0 | 0 | 327 |
| 43 | 327 | yes | 21 | 0 | 6754 | 20.7 | 21 | 0 | 2 | 0 | 0 |
| 44 | 327 | yes | 4 | 0 | 1025 | 3.1 | 4 | 0 | 1 | 0 | 0 |
| 45 | 327 | yes | 16 | 0 | 5044 | 15.4 | 16 | 0 | 2 | 0 | 0 |
| 46 | 327 | yes | 23 | 0 | 7111 | 21.7 | 23 | 0 | 1 | 0 | 0 |
| 47 | 327 | yes | 9 | 0 | 2696 | 8.2 | 9 | 0 | 4 | 0 | 0 |
| 48 | 326 | yes | 1 | 0 | 18 | 0.1 | 1 | 308 | 1 | 0 | 326 |
| 49 | 326 | yes | 1 | 0 | 134 | 0.4 | 1 | 192 | 1 | 0 | 93 |
| 50 | 326 | yes | 1 | 1 | 333 | 1.0 | 2 | 0 | 1 | 1 | 0 |
| 51 | 326 | yes | 0 | 4 | 1304 | 4.0 | 4 | 0 | 0 | 4 | 0 |
| 52 | 326 | yes | 1 | 1 | 331 | 1.0 | 2 | 0 | 1 | 1 | 0 |
| 53 | 326 | yes | 1 | 0 | 326 | 1.0 | 1 | 0 | 0 | 0 | 0 |
| 54 | 326 | yes | 19 | 0 | 6194 | 19.0 | 19 | 0 | 0 | 0 | 0 |
| 55 | 326 | yes | 10 | 0 | 3260 | 10.0 | 10 | 0 | 0 | 0 | 0 |
| 56 | 326 | yes | 11 | 0 | 3156 | 9.7 | 11 | 0 | 0 | 0 | 0 |
| 57 | 326 | yes | 1 | 0 | 187 | 0.6 | 1 | 139 | 1 | 0 | 0 |
| 58 | 326 | yes | 0 | 7 | 1346 | 4.1 | 6 | 0 | 0 | 3 | 326 |
| 59 | 326 | yes | 0 | 11 | 2734 | 8.4 | 11 | 0 | 0 | 4 | 326 |
| 60 | 326 | yes | 0 | 11 | 2734 | 8.4 | 10 | 0 | 0 | 1 | 326 |
| 61 | 326 | yes | 0 | 9 | 2838 | 8.7 | 9 | 0 | 0 | 5 | 0 |
| 62 | 326 | yes | 0 | 14 | 4317 | 13.2 | 14 | 0 | 0 | 6 | 0 |
| 63 | 326 | yes | 0 | 174 | 52615 | 161.4 | 174 | 0 | 0 | 3 | 0 |
| 64 | 72 | yes | 12 | 0 | 864 | 12.0 | 12 | 0 | 0 | 0 | 72 |
| 65 | 72 | yes | 4 | 0 | 288 | 4.0 | 4 | 0 | 0 | 0 | 0 |
| 66 | 72 | yes | 0 | 68 | 4896 | 68.0 | 68 | 0 | 0 | 1 | 0 |
| 67 | 70 | yes | 0 | 10 | 700 | 10.0 | 10 | 0 | 0 | 0 | 70 |
| 68 | 70 | yes | 0 | 87 | 6090 | 87.0 | 87 | 0 | 0 | 1 | 0 |
| 69-74 | 70 each | empty | 0 | 0 | 0 | 0 | 0 | all | 0 | 0 | all |
| 75-89 | 70 each | empty | 0 | 0 | 0 | 0 | 0 | all | 0 | 0 | 0 |
| 90-94 | 72 each | empty | 0 | 0 | 0 | 0 | 0 | all | 0 | 0 | all |
| 95 | 326 | yes | 0 | 8 | 2103 | 6.5 | 8 | 0 | 0 | 3 | 326 |
| 96 | 326 | yes | 0 | 18 | 5584 | 17.1 | 18 | 0 | 0 | 2 | 326 |
| 97 | 326 | yes | 0 | 2 | 652 | 2.0 | 2 | 0 | 0 | 0 | 326 |
| 98 | 326 | yes | 0 | 4 | 456 | 1.4 | 4 | 89 | 0 | 1 | 326 |
| 99 | 326 | yes | 0 | 8 | 2575 | 7.9 | 8 | 0 | 0 | 2 | 326 |
| 100 | 326 | yes | 0 | 257 | 83652 | 256.6 | 257 | 0 | 0 | 8 | 0 |

## Frame rate and GSD (none of these are measured)

| quantity | AIR-MOT README / paper | secondary sources (labelled) |
|---|---|---|
| fps | **Not stated** in the README. We could not read the paper body (IEEE paywall), so it is unverified there. | "5–10 FPS" for AIR-MOT, from Wan et al. 2024, [Sensors 24(19):6489](https://www.mdpi.com/1424-8220/24/19/6489). The Jilin-1 video satellites record "at a rate of 10 frames per second" ([Apollo Mapping](https://apollomapping.com/jilin-1-nighttime-satellite-imagery)). |
| GSD | **Not stated** in the README; paper unverified | "0.91 m to 1.27 m" (Wan et al. 2024, same link); "1–1.2 m" ([RingMo-Aerial, arXiv 2409.13366](https://arxiv.org/html/2409.13366v3)); Jilin-1 video at 92 cm and 1.2 m (Apollo Mapping) |
| resolution | not stated in README | 1920×1080 per RingMo-Aerial; **measured 1920×1080 everywhere [M]** |
| scenes | — | Abu Dhabi, Dubai, San Diego, Los Angeles, Sydney, Yokohama, Beijing, Sanya and Shanghai, Oct 2017 – Oct 2020 (Wan et al. 2024) |

The secondary sources give different video counts for AIR-MOT: 59 (Wan et al.) and 152 (RingMo-Aerial). Neither number matches this 100-video release.

## Split

**No official train/test split is published.** The README, repo files and issues give none. Issue [#5](https://github.com/HeQibin/TGraM/issues/5) claims the original 70-video folder held "only the training set"; the folder is now dead, so this cannot be checked. In AIR-MOT-100, the 31 videos with empty gt might be a withheld test set, but nothing states this. Any split we use is ours to define.

**TALOS split (frozen 2026-11-11, [`configs/splits/airmot.json`](../../configs/splits/airmot.json)).** The 69 annotated videos are split by video, 47 / 11 / 11 for train / val / test, with seed 0. The split is stratified so every part gets its share of the rare cases: videos with an aircraft moving more than 100 px (7 / 2 / 2), other aircraft videos (15 / 3 / 3), and ship-only videos (25 / 6 / 6). Test holds 35 aircraft tracks (4 moving more than 20 px) and 69 ship tracks (15 moving). Tracking is reported per class. The threat-warning evaluation uses both classes, because aircraft alone have too few movers.

**Decision (2026-11-11):** the project owner approved AIR-MOT-100 as the primary dataset despite the unverified uploader and the missing license. VISO remains the cross-dataset check. Every result citing AIR-MOT names this release.

**Frame rate used:** 10 fps, assumed (see above). Any number in seconds or m/s inherits this assumption and says so.

## Anomalies [M]

- **31/100 videos have no annotations** (listed above).
- **Black bands at the bottom of frames:** 6,519 frames in 32 videos have ≥ 8 pure-black bottom rows; 5,453 of those frames are in annotated videos. A typical band is about 392 rows, so the usable area is about 1920×688. In video 48, frames 228 onward are entirely black (99 frames). No gt box overlaps a band. This matches issue [#10](https://github.com/HeQibin/TGraM/issues/10).
- **6 exact duplicate rows:** video 12, ship id 113, frames 100-102, each row appears twice.
- **Track gaps (ships only, 10 tracks; aircraft have none):** v13#9, v14#1 (max gap 140 frames), v15#8, v16#21, v16#26, v16#27, v21#7 (104), v58#5, v60#1 (118), v63#57. The long gaps could be occlusion or a reused ID. We did not check them visually.
- IDs repeat across classes in the same video (v50 and v52, id 1), so tracks must be keyed by (class, id).
- Stray `63/img/Thumbs.db`. All 22,668 JPEGs decode.

## Polaris paths

- Data: `~/talos/data/raw/airmot/AIR-MOT-100/`; zip kept in `~/talos/data/raw/airmot/zip/`
- Script: `~/talos/data/scripts/inspect_airmot.py`
- Outputs: `~/talos/data/raw/airmot/stats/{summary.json,per_video.csv,tracks.csv,frames.csv}`
