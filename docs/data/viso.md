# VISO — data card (MOT subset, plane focus)

Role in TALOS: cross-dataset test set and fallback primary set for the aircraft tracker.

**[M]** = measured on Polaris from the downloaded files (`~/talos/data/scripts/inspect_viso.py`, 2026-11).
**[S]** = stated by a source (linked). Anything not marked is a description of layout we observed.

Sources: [repo](https://github.com/QingyongHu/VISO) · [paper arXiv v1](https://arxiv.org/abs/2111.12960) (Yin et al., IEEE TGRS 2021) · [SatVideoDT 2022](https://satvideodt.github.io/) · License **CC BY-NC-SA 4.0** [S, repo]. Imagery: Jilin-1 (Chang Guang Satellite).

## Bottom line for an aircraft tracker

- The released MOT subset has **6 plane sequences, 9 plane tracks and 2,268 plane boxes over 1,913 frames** [M]. That is a very small test set: report per-track results, not only aggregates.
- Plane boxes are **13–69 px** (median 39 w × 41 h) [M]. They are bigger than VISO cars (median ~5–9 px) but still small.
- All 9 plane tracks move more than 100 px over their life [M]. The median track speed is **~1.1–1.3 px/frame**, and the fastest is ~8 px/frame [M].
- **This is not the 100-sequence SatVideoDT set.** The Drive archive holds the paper's **47-sequence** MOT release [M]. The challenge's 100 sequences (videos 1–80 train, 81–90 val, 91–100 test) were a separate download ([challenge Drive folder](https://drive.google.com/drive/folders/1iRg72Bre4QagbqxcYevcgDoDV5oR-c23)). We have **not** downloaded them.
- The same plane frames reappear in the detection and SOT subsets, so do not train on VISO-detection plane images and then test on VISO-MOT planes.

## Download

| item | value |
|---|---|
| file | `VISO.zip` (Drive id `11G0pqEMletzPtueGbgD-Pq9stQAcvpWw`), via `uvx gdown`, one attempt, no quota block |
| size | **9,199,661,419 B** (matches Drive `content-length`) [M]; md5 `d69947bbfc2273d27037ec0404d8ef32` [M] |
| contents | 4 RAR4 archives (`Rar!\x1a\x07\x00`) inside the zip [M] |
| on disk | 20,759,334,341 B ≈ 20 GB total under `~/talos/data/raw/viso/` [M] = zip 9.20 GB + unzipped rars 9.21 GB + extracted `mot/` 2.35 GB |
| extracted | only `mot.rar` (16,278 files, all matching the archive listing) [M]. SOT and detection are listed, not extracted. |

Polaris had no `unrar` or `7z`. Extraction uses the system libarchive 3.3.3 through `libarchive-c` (`~/talos/data/scripts/rar_tool.py`). RAR4 is supported; RAR5 would not be.

| archive | entries | uncompressed bytes [M] |
|---|---|---|
| `VISO/mot.rar` | 16,424 | 2,337,407,714 |
| `VISO/SOT.rar` | 20,014 | 2,296,740,611 |
| `VISO/Detection_coco_format.rar` | 33,043 | 2,680,946,517 |
| `VISO/Detection_voc_format.rar` | 33,047 | 2,536,747,128 |

## Layout

```
~/talos/data/raw/viso/
  VISO.zip                      original download
  VISO/{mot,SOT,Detection_coco_format,Detection_voc_format}.rar
  listings/*.tsv                full file listing of each rar (path, bytes, is_dir)
  extracted/mot/<class>/<seq>/
      img/000001.jpg ...        6-digit, 1-based, contiguous, JPEG
      gt/gt.txt                 (car/* also has gt/gt_centroid.txt)
~/talos/data/stats/viso/        viso_mot_sequences.csv, viso_tracks_plane_ship.csv, viso_summary.json, inspect_out.txt
~/talos/data/scripts/           inspect_viso.py, rar_tool.py
```

Subsets in the release [M, from the listings]:

- **mot/**: 47 sequences, grouped by class folder: `car/001–038`, `plane/039–044`, `ship/045, 047`, `train/046`.
- **sot/**: same sequence ids and class folders. One gt file per target, named `<target>_<firstframe>_<lastframe>.txt`. Plane: 8 files, for example `039/1_1_180.txt`, `042/2_79_154.txt` and `044/2_385_500.txt`, and they match the MOT plane tracks. There is no SOT file for plane 041.
- **coco/**: per class, `train2017/val2017/test2017` jpgs + per-image xml + `instances_*.json`. Plane: train 1,005 / val 300 / test 608 = 1,913 images [M]. That equals the MOT plane frame count. Ship and train have no val split.
- **voc/**: per class, `JPEGImages`, `Annotations` and `ImageSets/Main/{train,val,trainval,test}.txt`. Plane: 1,913 images [M].

## MOT annotation format

There is **no class column**. Class is encoded only by the parent folder (`mot/plane/...`). Every track inside a sequence has that folder's class. There is no `seqinfo.ini` and no split file.

**Two formats coexist** [M], so a parser must handle both:

| format | sequences | line shape |
|---|---|---|
| A, comma, LF | car 001–027, plane 040, 043, ship 047, train 046 (31 sequences) | `frame,id,x_left,y_top,w,h,1,-1,-1,-1` |
| B, space, CRLF, sometimes a leading space | car 028–038, plane 039, 041, 042, 044, ship 045 (16 sequences) | `frame id x1 y1 x2 y2 -1 -1 -1 -1` |

Raw lines, format A (`car/001`):
```
1,1,582,24,6,9,1,-1,-1,-1
1,2,643,231,5,7,1,-1,-1,-1
1,3,664,278,4,6,1,-1,-1,-1
1,4,679,321,5,6,1,-1,-1,-1
1,5,678,330,5,6,1,-1,-1,-1
```
Columns: 1 = frame (1-based, matches `%06d.jpg`) · 2 = track id (unique within a sequence) · 3–4 = top-left x, y in px · 5–6 = box width, height in px · 7 = always `1` (MOTChallenge "consider" flag) · 8–10 = `-1` (unused 3D world coordinates).
`gt_centroid.txt` (cars only) is the same data, with columns 3–4 replaced by the box center (`585 = 582 + 6/2`, `28.5 = 24 + 9/2`).

Raw lines, format B (`plane/039`; lines end in `\r\n`):
```
 1 1 1301 569 1332 600 -1 -1 -1 -1
 1 2 ...
```
Columns: frame, id, **x1, y1, x2, y2** (corners), then `-1 ×4`. Column 7 is `-1`, not `1`.
We inferred the corner reading from the data; no source states it. In all 16 format-B files, x2 > x1 and y2 > y1 on every row [M]. Reading columns 5–6 as w,h would give planes 1,332 px wide inside a 1,454 px frame. The converted sizes (~30 px) match the format-A planes. Whether x2/y2 are inclusive (+1) is **unknown**.

Data-quality flags [M]:
- 4 duplicated (frame, id) rows: plane 039 id 1 at frames 5 and 129, and ship 045 id 2 at frames 8 and 87. In each, the second box is far from the track (for example 350 px away), so these are mislabeled ids. Drop or repair them before computing IDF1.
- 172 boxes with w or h ≤ 0 (cars 016, 017, 019, 020, 022, 025, 026, 027) and 244 boxes extending outside the image (cars 002, 016, 017, 018, 020, 025). There are none in plane sequences.
- `car/002/gt/gt.txt` ends with 1 blank line.
- Frame gaps inside tracks: plane 039 id 2 (3 missing frames), 041 id 1 (1), 042 id 2 (3), 044 id 2 (12) and ship 045 id 3 (77).
- Sequences 028–047 are mostly 1–3-track crops rather than full 1024² tiles (see resolutions below). They look like SOT-style clips stored in the MOT folder.

## Per-sequence table [M]

Resolution is H×W, checked on the first, middle and last frame of each sequence. All frames are 3-channel JPEG.

| class | seq | frames | H×W | gt fmt | tracks | boxes | w med / max | h med / max |
|---|---|---|---|---|---|---|---|---|
| car | 001 | 260 | 1025×1025 | A | 186 | 31,846 | 5 / 10 | 5 / 9 |
| car | 002 | 326 | 1024×1024 | A | 89 | 11,209 | 8 / 27 | 7 / 20 |
| car | 003 | 325 | 1024×1024 | A | 89 | 16,208 | 5 / 9 | 6 / 11 |
| car | 004 | 325 | 1024×1024 | A | 156 | 17,809 | 5 / 10 | 5 / 11 |
| car | 005 | 326 | 1024×1024 | A | 140 | 21,322 | 5 / 24 | 5 / 26 |
| car | 006 | 326 | 1024×1024 | A | 71 | 9,368 | 9 / 23 | 7 / 16 |
| car | 007 | 326 | 1024×1024 | A | 125 | 15,897 | 5 / 11 | 6 / 18 |
| car | 008 | 326 | 1024×1024 | A | 91 | 10,487 | 5 / 12 | 6 / 30 |
| car | 009 | 300 | 1024×1024 | A | 27 | 4,198 | 16 / 29 | 15 / 27 |
| car | 010 | 326 | 1024×1024 | A | 151 | 20,178 | 7 / 28 | 5 / 16 |
| car | 011 | 325 | 1024×1024 | A | 103 | 21,565 | 7 / 12 | 7 / 12 |
| car | 012 | 325 | 1024×1024 | A | 177 | 31,705 | 4 / 14 | 4 / 16 |
| car | 013 | 326 | 1024×1024 | A | 298 | 42,300 | 6 / 22 | 5 / 28 |
| car | 014 | 703 | 1024×1024 | A | 71 | 28,337 | 8 / 19 | 6 / 16 |
| car | 015 | 300 | 1024×1024 | A | 49 | 8,876 | 16 / 27 | 15 / 27 |
| car | 016 | 730 | 1024×1024 | A | 268 | 102,396 | 5 / 16 | 7 / 30 |
| car | 017 | 350 | 1024×1024 | A | 179 | 41,500 | 8 / 28 | 8 / 29 |
| car | 018 | 210 | 1024×1024 | A | 144 | 23,143 | 9 / 27 | 8 / 29 |
| car | 019 | 170 | 1024×1024 | A | 126 | 17,297 | 9 / 30 | 8 / 29 |
| car | 020 | 750 | 1024×1024 | A | 147 | 49,828 | 7 / 26 | 5 / 21 |
| car | 021 | 480 | 1024×1024 | A | 86 | 32,645 | 5 / 8 | 5 / 8 |
| car | 022 | 481 | 1024×1024 | A | 201 | 78,698 | 5 / 22 | 4 / 20 |
| car | 023 | 480 | 1024×1024 | A | 61 | 22,575 | 5 / 16 | 4 / 8 |
| car | 024 | 481 | 1024×1024 | A | 212 | 79,682 | 5 / 16 | 4 / 15 |
| car | 025 | 480 | 1024×1024 | A | 113 | 34,915 | 5 / 17 | 6 / 28 |
| car | 026 | 371 | 1024×1024 | A | 157 | 34,272 | 8 / 18 | 7 / 15 |
| car | 027 | 332 | 1024×1024 | A | 123 | 29,432 | 8 / 160 | 8 / 86 |
| car | 028 | 200 | 1348×1348 | B | 1 | 200 | 6 / 8 | 15 / 17 |
| car | 029 | 200 | 1348×1348 | B | 1 | 200 | 19 / 23 | 7 / 18 |
| car | 030 | 200 | 1348×1348 | B | 1 | 197 | 20 / 26 | 9 / 22 |
| car | 031 | 200 | 674×620 | B | 1 | 200 | 12 / 19 | 16 / 21 |
| car | 032 | 200 | 674×620 | B | 1 | 200 | 10 / 24 | 7 / 17 |
| car | 033 | 300 | 230×280 | B | 1 | 300 | 6 / 8 | 12 / 15 |
| car | 034 | 300 | 230×280 | B | 1 | 300 | 6 / 8 | 5 / 12 |
| car | 035 | 299 | 223×220 | B | 1 | 299 | 8 / 14 | 7 / 10 |
| car | 036 | 358 | 285×543 | B | 1 | 358 | 5 / 7 | 4 / 7 |
| car | 037 | 346 | 285×543 | B | 1 | 346 | 6 / 10 | 7 / 9 |
| car | 038 | 358 | 285×543 | B | 1 | 358 | 8 / 11 | 5 / 8 |
| **plane** | **039** | 180 | 750×1454 | B | **2** | 359 | 30 / 34 | 30 / 35 |
| **plane** | **040** | 250 | 489×701 | A | **1** | 250 | 65 / 69 | 63 / 67 |
| **plane** | **041** | 325 | 521×1283 | B | **1** | 324 | 30 / 38 | 23 / 32 |
| **plane** | **042** | 300 | 600×900 | B | **2** | 373 | 38 / 44 | 42 / 46 |
| **plane** | **043** | 358 | 285×543 | A | **1** | 358 | 41 / 52 | 39 / 47 |
| **plane** | **044** | 500 | 1080×1024 | B | **2** | 604 | 53 / 68 | 56 / 61 |
| ship | 045 | 320 | 451×1345 | B | 3 | 884 | 33 / 56 | 17 / 36 |
| train | 046 | 250 | 286×247 | A | 1 | 247 | 31 / 56 | 50 / 58 |
| ship | 047 | 300 | 512×512 | A | 1 | 300 | 15 / 21 | 17 / 24 |

Totals [M]: 47 sequences · 16,204 frames · 3,665 tracks · 844,345 boxes.
By class: car 38 sequences / 3,651 tracks / 840,646 boxes / 13,421 frames; plane 6 / 9 / 2,268 / 1,913; ship 2 / 4 / 1,184 / 620; train 1 / 1 / 247 / 250.
Every track's class equals its folder's class, so the per-class track counts above are exact.
Paper v1 says "3711 tracklets with 47 sequences" [S]; we count 3,665 [M].

## Plane tracks [M]

Speed is measured at the box center. Rows from duplicated (frame, id) pairs are dropped for the speed columns. Mean speed = path length ÷ frame span. At 10 fps [S], 1 px/frame = 10 px/s.

| seq | id | frames | boxes | w×h med | net displacement px | mean speed px/frame | median step px/frame |
|---|---|---|---|---|---|---|---|
| 039 | 1 | 1–180 | 182 (4 dup rows) | 31×30 | 1,303 | 7.31 | 7.16 |
| 039 | 2 | 1–180 | 177 | 29×30 | 144 | 1.02 | 1.12 |
| 040 | 1 | 1–250 | 250 | 65×63 | 192 | 1.09 | 1.12 |
| 041 | 1 | 1–325 | 324 | 30×23 | 1,012 | 3.38 | 3.35 |
| 042 | 1 | 1–300 | 300 | 38×42 | 285 | 1.29 | 1.12 |
| 042 | 2 | 79–154 | 73 | 39×42 | 587 | 7.88 | 8.02 |
| 043 | 1 | 1–358 | 358 | 41×39 | 232 | 1.17 | 1.12 |
| 044 | 1 | 1–500 | 500 | 54×56 | 178 | 0.92 | 0.71 |
| 044 | 2 | 385–500 | 104 | 35×28 | 241 | 2.34 | 2.12 |

Summary [M]:
- Plane box size over all 2,268 boxes: **w min 16 / median 39 / p90 63 / max 69 px**; **h min 13 / median 41 / p90 61 / max 67 px**.
- Track length: 73–500 boxes (median 250).
- Net displacement: min 144, median 241, p90 1,070, max 1,303 px.
- Mean speed (de-duplicated): min 0.92, median 1.29, p90 7.43, max 7.88 px/frame.
- Without de-duplication, 039 id 1 reads 19.5 px/frame. That number comes from the duplicate rows, not real motion.
- Tracks moving > 20 px over their life: **9 / 9**. Tracks moving > 100 px: **9 / 9**. This holds for both net displacement and maximum distance from the start.
- Planes per sequence: 1–2, so these clips test track maintenance more than association in dense scenes.

## Ship tracks [M]

| seq | id | boxes | w×h med | net displacement px | mean speed (de-dup) px/frame |
|---|---|---|---|---|---|
| 045 | 1 | 320 | 21×10 | 242 | 2.00 |
| 045 | 2 | 322 (4 dup rows) | 43×17 | 186 | 1.05 |
| 045 | 3 | 242 (77-frame gap) | 33×23 | 99 | 1.49 |
| 047 | 1 | 300 | 15×17 | 142 | 1.22 |

Ship boxes over 1,184 boxes: w 11 / 26 / 43 / 56 and h 6 / 17 / 23 / 36 px (min / median / p90 / max). Tracks > 20 px: 4/4. Tracks > 100 px net: 3/4 (4/4 by maximum distance from the start).

## Official splits [S]

| set | train | val | test | source |
|---|---|---|---|---|
| Paper, MOT (this release) | videos 1–27 (3,159 tracklets) | 28–40 (440) | 41–47 (658) | arXiv v1 §III-B5 |
| Paper, SOT | videos 1–20 | 21–23 | 24–27 | arXiv v1 §III-B5 |
| Paper, detection | 13,470 images | 535 | 3,725 | arXiv v1 §III-B5 |
| SatVideoDT 2022 Track 3 (MOT) | videos 1–80 | 81–90 | 91–100 | satvideodt.github.io |
| Release COCO split, plane class | 1,005 images | 300 | 608 | [M] from the archive folders |

Under the paper's MOT split, **every plane sequence (039–044) falls in validation (28–40) or test (41–47)**: 039 and 040 in validation, 041–044 in test. No plane sequence is in train.
Inconsistencies in the paper: its tracklet counts sum to 4,257, not the stated 3,711, and the paper says 47 videos while the repo README says 40.
The SatVideoDT numbering (1–100, 32,825 frames [S]) is a different numbering from this release, and the two cannot be mapped from what we have.

## Frame rate and GSD [S]

- **Frame rate: 10 fps.** Paper: "the video frame rate is 10 frames per second" (arXiv v1 §III-A1). The same paper's Table II lists Jilin-1 Frequency 10 Hz.
- **GSD: 0.92 m**, only as the "Spatial Resolution" of JiLin-1 in the paper's satellite-spec table (Table II). The paper does not say this is the GSD of the released crops, and no per-video GSD is given. Treat it as approximate; true per-video GSD is **unknown**.
- At 0.92 m/px (approximate), median plane boxes of ~39–41 px are about 36–38 m. That is a derived estimate, not a measurement.

## Anything odd

1. The archive is the 47-sequence paper release, not the 100-sequence challenge set. Plane test material is only 9 tracks.
2. There are two gt formats, and format B's corner semantics are inferred, not documented.
3. MOT, SOT, COCO and VOC all contain the same 1,913 plane frames. There is no held-out plane data inside VISO.
4. Image sizes vary from 220 to 1454 px per side. Sequence 001 is 1025×1025.
5. One sequence (039) has duplicate-id label errors that would penalize a correct tracker.
