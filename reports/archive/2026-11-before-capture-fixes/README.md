# Reports of the race before the capture fixes

The 10 lap race on CompetitionMap3 as it was reported until 14 November 2026: fine-tuned YOLO26s in half precision
through `torch.compile`, the same pipelines as now, but recorded before three changes to how the images get out of
the simulator: the simulator ran without `TCP_NODELAY` (about one image in eight came 40 ms late), Fast DDS cut every
image into 57 fragments, and stereo only paired images less than 12 ms apart. Stereo was capped at 10 m. The bag
(`runs/live_competitionmap3_10laps_before_fixes`) is no longer kept.

`docs/notes/simulator-notes.md` ("Camera capture") has the measurements behind the fixes.
