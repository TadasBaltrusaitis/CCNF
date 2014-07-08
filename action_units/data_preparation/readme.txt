Scripts in this folder extract the features from DISFA dataset that can be used for AU recognition.

These features are of two kinds:
1. patches of [patch_size x patch_size] around facial landmarks of interest extracted from a similarity normalised face. The face size is controlled by the face_scale parameter.
2. non-rigid shape parameters

To extract the features use extract_disfa_features(patch_size, face_scale) function that will return the features, together with the video id's for experiments.

The current patch_size used for experiments is 24 and face_scale 0.75.

After data preparation head to eccv_exps folder