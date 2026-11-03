  var PHASES = [
    {
      title: "Data and setup", from: 1, to: 2, status: "done",
      req: [
        "Get the OpenGraph and HOT3D ground truth data",
        "Fix the tuning / held-out split",
        "Check SAM 3.1 and Muse Spark calls and reply parsing"
      ],
      chk: [
        ["done", "Build the work environment, check robot hand retargeting"],
        ["done", "Collect 72 OpenGraph episodes, split by episode"],
        ["done", "Collect HOT3D ground truth (hand and object poses, object 3D models)"],
        ["done", "Fix the split (OpenGraph 50/22, HOT3D 6/3 persons)"],
        ["done", "Check SAM 3.1 and Muse Spark replies"],
        ["done", "Check contact label reliability (tactile threshold 800)"],
        ["done", "Pick the SAM object prompt (small object)"]
      ]
    },
    {
      title: "Contact cutting · Test 1", from: 3, to: 4, status: "done",
      req: [
        "Find each hand's contact start and end from video alone",
        "Boundary F1 (±5, ±15 frames) and contact IoU against the tactile glove",
        "Compare A Muse Spark alone, B SAM rule, C B plus a Muse Spark boundary check on the same terms"
      ],
      chk: [
        ["done", "Run the whole pipeline on 3 pilot episodes (EXP-06)"],
        ["done", "Fix the measurement plan (3 compared methods)"],
        ["done", "Fix the main ground truth: tactile glove with flicker cleaned"],
        ["done", "Build method B and the reference rows (EXP-14)"],
        ["done", "Add cut on object change and wait options to B (EXP-22, EXP-24)"],
        ["done", "Real A calls on tuning: Muse Spark 7 frames early, shift doubles F1 (EXP-29)"],
        ["done", "One A prompt revision: no gain, version 1 stays (EXP-30)"],
        ["done", "Fix B: F1 ±5 0.153, IoU 0.683 (50 tuning episodes, EXP-32)"],
        ["done", "Run A, B and C on 22 held-out episodes (EXP-33, EXP-34, EXP-35)"],
        ["done", "Score the three methods on 22 held-out: A plus shift F1 ±5 0.195, IoU 0.850 (EXP-37)"]
      ]
    },
    {
      title: "HOT3D ground truth labels", from: 5, to: 6, status: "done",
      req: [
        "Move the mocap hand onto the Allegro robot hand and judge each segment copyable or not",
        "Copyable rate of 20 to 80% on the tuning set",
        "Both hands"
      ],
      chk: [
        ["done", "Get fingertip 3D positions (UmeTrack)"],
        ["done", "Collect video of 136 sequences with ground truth"],
        ["done", "Cut Test 2 clips (2,385 tuning, 1,426 held-out)"],
        ["done", "First label: fingertip to surface distance, 54% on tuning (EXP-10)"],
        ["done", "Label method check: all three methods fail the rule, labels on hold (EXP-15)"],
        ["done", "Fix a new definition: position fit, link mesh collision, contact reach"],
        ["done", "Tuning v2 labels: 47%, hand size correlation +0.04, passes (EXP-16)"],
        ["done", "Held-out v2 labels: 46.4%, 43 to 51% per person (EXP-20)"]
      ]
    },
    {
      title: "Score learning · Test 2", from: 7, to: 7, status: "done",
      req: [
        "12 item grades and a 0 to 100 score per clip",
        "Logistic regression on 400 tuning clips, weights published",
        "AUC on 1,426 held-out clips: our score vs Muse Spark alone, object baseline, Muse Spark plus 12 items",
        "Soft objects get no score and go to a touch check list"
      ],
      chk: [
        ["done", "AUC of the three video items alone: no signal (EXP-12)"],
        ["done", "Measured again with the v2 label: object 0.74, video items add about 0.01 (EXP-17)"],
        ["done", "Direction: grow the object items (geometry alone 0.695, EXP-18)"],
        ["done", "Six Muse Spark object items with high, mid, low rules: stand-in grades 0.683 (EXP-19)"],
        ["done", "SAM and Muse Spark pilot: grasped object 99%, grade match 27/30 (EXP-23)"],
        ["done", "Muse Spark on 400 tuning clips: gate weak (0.636, 0.601), go to held-out (EXP-29)"],
        ["done", "One checklist prompt revision: no gain, version 1 stays (EXP-30)"],
        ["done", "Collect SAM and Muse Spark answers for 1,426 held-out clips (EXP-38, EXP-39)"],
        ["done", "Held-out AUC: our score 0.624, Muse Spark alone 0.501, object name 0.771 (EXP-40)"],
        ["done", "Unseen objects 0.571, within an object 0.506 (EXP-40)"],
        ["done", "Depth limit check: holds at 10 mm, chance at 2 mm (EXP-41)"],
        ["done", "Soft object labels: 81.7% agreement, kappa 0.613, 4 objects left out"],
        ["done", "Soft object filter check: 51.7% of soft held back, score scoped to rigid objects (EXP-42)"]
      ]
    },
    {
      title: "Viewer · demo", from: 8, to: 9, status: "done",
      req: [
        "A viewer with segments ranked by score, boxes on the video and a contact signal chart",
        "Demo video and README"
      ],
      chk: [
        ["done", "Viewer: clip score tab (top and bottom 15 held-out scores), contact timing tab (22 held-out, glove vs three methods)"],
        ["done", "Three presets (strict 70+, balanced 50+, volume 30+)"],
        ["done", "README (main results, how to run)"],
        ["done", "Demo video plan: 3 minutes, 6 scenes, viewer recordings and 4 figures for the video"],
        ["done", "Recording setup: direct viewer links, two clips side by side"],
        ["done", "3 minute demo video (English, Korean)"]
      ]
    },
    {
      title: "Final measurement", from: 10, to: 10, status: "done",
      req: [
        "Measure the final numbers once on the held-out set"
      ],
      chk: [
        ["done", "Final held-out measurement (EXP-37, EXP-40, EXP-42)"],
        ["done", "Numbers in the report and README"],
        ["done", "Ground truth packed as a benchmark: label files and independent scorers (EXP-28)"]
      ]
    }
  ];

  var STATUS_TXT = { done: "Done", doing: "In progress", todo: "Planned" };

