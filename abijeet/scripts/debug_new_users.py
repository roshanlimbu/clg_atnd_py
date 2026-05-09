"""
Diagnostic script to identify why new users are not being stored.

Run this to test:
1. If embedding computation is working
2. If database storage is working
3. If face_signature is being set correctly
"""

import logging
from pathlib import Path
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATABASE_FILE = BASE_DIR / "attendance.db"


def test_embedding_computation():
    """Test if we can compute embeddings from images."""
    print("\n" + "="*60)
    print("TEST 1: Embedding Computation")
    print("="*60)
    
    try:
        from face_identity import FaceIdentityManager
        from database import DatabaseManager
        
        db = DatabaseManager(DATABASE_FILE)
        identity_mgr = FaceIdentityManager(db)
        
        # Create a test face image (just a random image)
        # This will likely fail since it's not a real face, but it shows the code path
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        print(f"✓ Created test image: shape={test_image.shape}")
        
        # Try to compute embedding
        embedding = identity_mgr._compute_embedding(test_image)
        
        if embedding is None:
            print(f"✗ Embedding is None - this is EXPECTED for random image")
            print(f"  This means _compute_embedding() returns None for non-face images")
        else:
            print(f"✓ Embedding computed: shape={embedding.shape}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_database_storage():
    """Test if new persons are being stored in database."""
    print("\n" + "="*60)
    print("TEST 2: Database Storage")
    print("="*60)
    
    try:
        from database import DatabaseManager
        import json
        from datetime import date
        
        db = DatabaseManager(DATABASE_FILE)
        
        # Try to add a test person
        test_person_id = f"TEST_PERSON_{int(date.today().timestamp())}"
        test_signature = json.dumps([[0.1, 0.2, 0.3] + [0.0]*125])  # Mock 128-D embedding
        
        print(f"✓ Adding test person: {test_person_id}")
        print(f"  Signature length: {len(test_signature)}")
        
        result = db.add_person(
            person_id=test_person_id,
            face_signature=test_signature,
            display_name="Test Person",
        )
        
        if result:
            print(f"✓ Test person added to database")
            
            # Try to retrieve it
            all_persons = db.get_all_persons()
            print(f"✓ Total persons in database: {len(all_persons)}")
            
            # Check if our test person is there
            found = False
            for person in all_persons:
                if person['person_id'] == test_person_id:
                    found = True
                    print(f"✓ TEST PERSON FOUND in database!")
                    print(f"  - person_id: {person['person_id']}")
                    print(f"  - face_signature length: {len(person['face_signature']) if person['face_signature'] else 'NULL'}")
                    print(f"  - display_name: {person['display_name']}")
                    break
            
            if not found:
                print(f"✗ TEST PERSON NOT FOUND after insertion!")
                print(f"  This could mean:")
                print(f"  1. Database file is different than expected")
                print(f"  2. face_signature is being set to NULL")
                print(f"  3. Transaction is not being committed")
        else:
            print(f"✗ Failed to add test person to database")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_face_identity_new_user_flow():
    """Test the complete flow of creating a new user."""
    print("\n" + "="*60)
    print("TEST 3: New User Creation Flow (with Mock Embedding)")
    print("="*60)
    
    try:
        from face_identity import FaceIdentityManager
        from database import DatabaseManager
        import numpy as np
        
        db = DatabaseManager(DATABASE_FILE)
        identity_mgr = FaceIdentityManager(db)
        
        # Create a mock embedding directly (bypassing embedding computation)
        mock_embedding = np.random.randn(128).astype(np.float64)
        
        print(f"✓ Created mock embedding: shape={mock_embedding.shape}")
        
        # Manually call _create_person() with the mock embedding
        person_id = identity_mgr._create_person(
            embedding=mock_embedding,
            pose_label="front",
            pose_angles={"yaw": 0, "pitch": 0, "roll": 0}
        )
        
        print(f"✓ New person created: {person_id}")
        
        # Check if the person is stored in memory
        found_in_memory = any(kf.person_id == person_id for kf in identity_mgr._known)
        print(f"  - Found in memory: {found_in_memory}")
        
        # Check if the person is in database
        all_persons = db.get_all_persons()
        found_in_db = any(p['person_id'] == person_id for p in all_persons)
        print(f"  - Found in database: {found_in_db}")
        
        # Get the person from database to verify face_signature
        for p in all_persons:
            if p['person_id'] == person_id:
                sig_present = p['face_signature'] is not None and len(p['face_signature']) > 0
                print(f"  - face_signature present: {sig_present}")
                if sig_present:
                    print(f"    (length: {len(p['face_signature'])} chars)")
                break
        
        if found_in_db:
            print(f"✓ TEST PASSED: New user stored successfully!")
        else:
            print(f"✗ TEST FAILED: New user not found in database!")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def check_database_schema():
    """Check if persons table has correct schema."""
    print("\n" + "="*60)
    print("TEST 4: Database Schema")
    print("="*60)
    
    try:
        from database import DatabaseManager
        
        db = DatabaseManager(DATABASE_FILE)
        
        # Get table info
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(persons)")
            columns = cursor.fetchall()
            
            print(f"✓ Persons table schema:")
            for col in columns:
                cid, name, type_, notnull, dflt_value, pk = col
                print(f"  - {name}: {type_} (PK: {bool(pk)}, NOT NULL: {bool(notnull)})")
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def check_attendance_photos():
    """Check if attendance photos are being saved for new users."""
    print("\n" + "="*60)
    print("TEST 5: Attendance Photo Storage")
    print("="*60)
    
    try:
        from pathlib import Path
        from datetime import date
        
        photos_dir = BASE_DIR / "attendance_photos"
        
        if not photos_dir.exists():
            print(f"✗ Attendance photos directory not found: {photos_dir}")
            return
        
        print(f"✓ Photos directory exists: {photos_dir}")
        
        # Check for today's photos
        today = date.today()
        today_dir = photos_dir / str(today)
        
        if today_dir.exists():
            photos = list(today_dir.glob("*.jpg"))
            print(f"✓ Photos for today: {len(photos)}")
            if photos:
                for photo in photos[:5]:  # Show first 5
                    print(f"  - {photo.name}")
        else:
            print(f"ℹ No photos for today yet: {today_dir}")
        
        # Check overall statistics
        all_dates = [d for d in photos_dir.iterdir() if d.is_dir()]
        print(f"✓ Total date directories: {len(all_dates)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║   DIAGNOSTIC: Why New Users Aren't Being Stored           ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Run all tests
    check_database_schema()
    test_database_storage()
    test_embedding_computation()
    test_face_identity_new_user_flow()
    check_attendance_photos()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("""
If TEST 3 passed but you're still not seeing new users:
  1. Check that face_identity._compute_embedding() is returning None
     → This would cause identify() to return "Unknown"
  2. Check that the aligned_image crop is too small or invalid
     → Increase face detector crop size
  3. Check camera feed is providing good quality faces
     → Test with upload_reference_images or real faces
    """)
