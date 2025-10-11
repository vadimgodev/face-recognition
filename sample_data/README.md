# Sample Data

This directory is used for test face images. It is excluded from version control via `.gitignore` to prevent accidental commit of personal biometric data.

## Directory Structure

```
sample_data/
├── alice.jpg                  # Enrollment photo for "Alice"
├── alice_test.jpg             # Test photo for recognition
├── bob.jpg                    # Enrollment photo for "Bob"
├── bob_test.jpg               # Test photo for recognition
├── unprocessed_face_images/   # For batch enrollment script
├── processed_face_images/     # Successfully enrolled (auto-moved)
└── failed_face_images/        # Failed enrollment (auto-moved)
```

## Batch Enrollment Filename Format

For use with `scripts/batch_enroll.py`, images should follow this naming convention:

```
{room_id}_{user_name}_{location}_{timestamp}.jpg
```

Examples:
- `101_john_lobby_20250101120000.jpg`
- `202_jane_entrance_20250315143000.jpg`

## Public Datasets for Testing

If you need face images for testing, consider these publicly available datasets:

- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) -- 13,000+ face images
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) -- 200,000+ celebrity face images
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) -- Face detection benchmark

**Important:** Always verify the license and terms of use for any dataset before using it. Never commit real face photos to the repository.
