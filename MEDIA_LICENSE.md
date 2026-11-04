# Dataset media in this repository and on the site

The code and the reports are our own work. Some images and video clips in
this repository, on the project site and in the demo videos are cut from two
public datasets. This page lists them, gives each source and license, and
says what we changed. It follows the license files that come with the data.
It is not a legal review.

## Sources

**HOT3D** by Meta Platforms Technologies, LLC,
<https://www.projectaria.com/datasets/hot3D/>, under the HOT3D Dataset License
Agreement (last updated May 29, 2024):

- Sequence data (the Project Aria video and the annotations other than hands):
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Hand annotations: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- 3D object models: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
  with the added term that they may not be sold or put into a product that
  is sold

**OpenGraph Ego-Tactile Manipulation** by OpenGraph Labs,
<https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation>,
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## What we share, and under which license

| Files | Content | Changes we made | License of these files |
| --- | --- | --- | --- |
| `web/assets/try/*.mp4` | 6 HOT3D video clips (Try page) | cut in time, resized, re-encoded | CC BY-SA 4.0 |
| `web/label-media/hot3d_*.jpg`, `hot3d_*.mp4` | 33 HOT3D frames and clips (password protected labeling page) | cut in time, resized, re-encoded | CC BY-SA 4.0 |
| `web/label-media/opengraph_*.jpg`, `opengraph_*.mp4` | 71 OpenGraph frames and clips (labeling page) | cut in time, resized, re-encoded | CC BY 4.0 |
| `web/assets/video/*.mp4`, `graspgrade_demo_poster.jpg` | demo videos with HOT3D and OpenGraph clips | cut, resized, combined with our charts, captions and voice | CC BY-SA 4.0 |
| `web/assets/clip.mp4`, `web/assets/frame.jpg` and the other figures with OpenGraph frames | OpenGraph frames, some with SAM 3.1 masks or our timelines drawn on them | cut, resized, drawings and text added | CC BY 4.0 |
| Figures with HOT3D frames that draw no hand annotations: `exp19_object_sheet.jpg`, `exp23_sam_chosen_object.png` | HOT3D frames with object boxes or SAM masks | cut, resized, drawings and text added | CC BY-SA 4.0 |
| Figures with HOT3D frames that draw hand annotations or hands built from them: `exp08_*_orientation.jpg`, `exp09_*_hands.jpg`, `exp10_examples.jpg`, `exp15_can_grasps.jpg`, `exp16_examples.jpg`, `exp16_strip.jpg`, `exp25_fisheye_example.png` | HOT3D frames with hand points, hand boxes, robot hand points and object model points | cut, resized, drawings and text added | CC BY-NC-SA 4.0, and the object model parts may not be sold |
| `benchmark/number2/clips.csv` | labels built from HOT3D hand annotations, object poses and object models | see `benchmark/number2/LICENSE.txt` | CC BY-NC-SA 4.0, not for sale |
| `benchmark/number1/ground_truth.json` | contact intervals from the OpenGraph gloves | see `benchmark/number1/LICENSE.txt` | CC BY 4.0 |

Other parts of the demo videos: the robot in the first scene is our render
of the Apptronik Apollo model from MuJoCo Menagerie (Apache 2.0, made by
`web/tools/robot_render.py`). The English narration is an ElevenLabs voice,
credited on the end card and the demo page. The Korean narration is the
Windows built-in voice.

The figures sit in `docs/experiments/img/`, and copies sit in `web/assets/`,
`web/assets/logs/` and `paper/figures/`. Charts that show only numbers are
our own work.

The full datasets are not in this repository. Get them from the links above.
