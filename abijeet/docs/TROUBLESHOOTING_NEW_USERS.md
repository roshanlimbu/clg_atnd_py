# 🔍 Why New Users Aren't Being Stored - Diagnosis & Fix

## Quick Diagnosis (Run This First)

```bash
python debug_new_users.py
```

This will tell you exactly where the problem is. Follow the output to identify which test fails.

---

## 5 Most Common Reasons (In Order of Likelihood)

### ❌ **REASON 1: Face Embedding Computation Failing (MOST COMMON)**

**Symptoms:**
- New faces show as "UNKNOWN" instead of getting a new ID
- System doesn't create person IDs for new users
- `is_new: false` never appears in logs

**Root Cause:**
`identify()` returns early if `_compute_embedding()` returns `None`:

```python
def identify(self, face_image: np.ndarray) -> FaceIdentityResult:
    embedding = self._compute_embedding(face_image)
    if embedding is None:  # <-- EXITS HERE FOR NEW USERS
        return FaceIdentityResult(
            person_id="Unknown",
            confidence=0.0,
            unknown=True,
        )
```

**Why embedding computation fails:**
- Face crop from detector is too small (<20x20 pixels)
- Face crop is too low quality
- Face encoding fails (dlib can't find face inside crop)
- No face is actually in the cropped region

**Fix:**
Check if this is the issue:
```python
# In debug_new_users.py, look at TEST 1 output
# If it says "Embedding is None", this is your problem
```

**Solution:**

**Option A: Enable fallback encoding (RECOMMENDED)**
```python
# face_identity.py, line ~430
encodings = self._face_recognition.face_encodings(
    face_rgb,
    known_face_locations=locations,
    num_jitters=1,  # <-- CHANGE 0 to 1 for more robust encoding
    model="small",
)
```

**Option B: Improve face crop quality**
```python
# In detector.py, increase crop size
# Find: face_images.append(...) 
# Add padding around detected face
```

---

### ❌ **REASON 2: Face Signature Not Saved (NULL in Database)**

**Symptoms:**
- Person IDs are being created
- But when you restart the system, new users aren't recognized
- Database shows empty `face_signature` column for new users

**Root Cause:**
```python
def _create_person(self, embedding, pose_label=None, pose_angles=None):
    serialized = self._serialize([embedding]) if embedding is not None else None
    if self.db.add_person(person_id, serialized):  # <-- Passing NULL
```

If serialization fails or returns empty string, the face_signature is NULL.

**Check the database:**
```bash
sqlite3 attendance.db
SELECT person_id, face_signature, display_name FROM persons WHERE face_signature IS NULL;
```

If this returns rows, that's your problem.

**Solution:**
```python
# In face_identity.py, _create_person() method

# Before:
serialized = self._serialize([embedding]) if embedding is not None else None

# After (WITH VALIDATION):
if embedding is None:
    logger.error(f"Cannot create person: embedding is None!")
    return None

serialized = self._serialize([embedding])
if not serialized or serialized == "null" or len(serialized) < 100:
    logger.error(f"Serialization failed! serialized={serialized}")
    return None
```

---

### ❌ **REASON 3: Database Query Filter Excludes New Users**

**Symptoms:**
- System seems to store users, but they're never recognized again
- `get_all_persons()` doesn't return new users

**Root Cause:**
In `database.py`, the `get_all_persons()` query filters:
```sql
WHERE face_signature IS NOT NULL
```

If the face_signature is somehow NULL, empty, or malformed JSON, new users won't be loaded!

**Solution:**
```python
# database.py, line ~630

# Before:
cursor.execute("""
    SELECT person_id, face_signature, display_name, role, reference_photo_path, registered_date
    FROM persons
    WHERE face_signature IS NOT NULL
    ORDER BY person_id
""")

# After (DEBUGGING):
cursor.execute("""
    SELECT person_id, face_signature, display_name, role, reference_photo_path, registered_date
    FROM persons
    ORDER BY person_id
""")
# Remove the WHERE clause to see ALL persons
# Then in face_identity.py, add validation:
def _load_known_embeddings(self) -> List[KnownFace]:
    result = []
    for person in self.db.get_all_persons():
        sig = person["face_signature"]
        if not sig:  # Log NULL/empty signatures
            logger.warning(f"Person {person['person_id']} has NO face_signature!")
            continue
        embeddings = self._deserialize(sig)
        if not embeddings:  # Log failed deserialization
            logger.warning(f"Failed to deserialize {person['person_id']}: {sig[:50]}...")
            continue
        result.append(...)
    return result
```

---

### ❌ **REASON 4: Transaction Not Committed**

**Symptoms:**
- New users created during session
- After restart, users are gone
- Database file size not growing

**Root Cause:**
Database transaction isn't committed:
```python
conn.execute("INSERT INTO persons...")
# Missing: conn.commit()
```

**Check:**
```python
# database.py, add_person() method - look for conn.commit()
```

**Solution:**
The code already has `conn.commit()`, but verify:
```python
def add_person(self, person_id: str, face_signature: Optional[str] = None, ...):
    try:
        with self._get_connection() as conn:
            conn.execute("""INSERT INTO persons...""", (...))
            conn.commit()  # <-- MUST BE HERE
            logger.info(f"Person registered: {person_id}")
            return True
```

---

### ❌ **REASON 5: Face Crop Too Small or Corrupted**

**Symptoms:**
- Some faces are stored, some aren't
- Works for close-up faces, fails for distant faces
- Random "Unknown" results

**Root Cause:**
YOLO provides bounding box, but crop might be:
- Too small (< 50x50)
- Mis-aligned (eyes not in center)
- Compressed/distorted

**Solution:**
```python
# In detector.py or face_identity.py, add validation:

def identify(self, face_image: np.ndarray) -> FaceIdentityResult:
    # NEW: Validate image quality
    h, w = face_image.shape[:2]
    if h < 50 or w < 50:
        logger.debug(f"Face too small: {w}x{h}, skipping")
        return FaceIdentityResult(
            person_id="Unknown",
            confidence=0.0,
            unknown=True,
        )
    
    # Continue with normal flow...
    embedding = self._compute_embedding(face_image)
```

---

## Step-by-Step Verification

### Step 1: Run Diagnostic
```bash
python debug_new_users.py
```

### Step 2: Check Database Directly
```bash
sqlite3 attendance.db
SELECT COUNT(*) as total_persons FROM persons;
SELECT COUNT(*) as with_signature FROM persons WHERE face_signature IS NOT NULL;
SELECT COUNT(*) as without_signature FROM persons WHERE face_signature IS NULL;
.exit
```

### Step 3: Enable Debug Logging
```python
# main.py, around line 100
logging.basicConfig(
    level=logging.DEBUG,  # <-- CHANGE from INFO to DEBUG
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"attendance_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler(),
    ],
)
```

Then run: `python main.py` and watch for:
- `Registered new face ID: XXXXX` (new user created)
- `Person already exists` (person already in DB)
- `Head pose detection failed` (pose detection issue)
- `Could not allocate a unique face ID` (database error)

### Step 4: Test With Known Good Face
```bash
python pose_aware_matching_example.py
# Select option 1: Live attendance with profiles
```

Approach the camera slowly from different angles. If this works but main.py doesn't, the issue is in the preprocessing pipeline.

---

## Quick Fixes (Try These First)

### Fix 1: Enable More Robust Face Encoding
```python
# face_identity.py, line ~435
# Change num_jitters from 0 to 1
encodings = self._face_recognition.face_encodings(
    face_rgb,
    known_face_locations=locations,
    num_jitters=1,  # ← Change this
    model="small",
)
```

### Fix 2: Increase Face Crop Padding
```python
# detector.py or camera.py, where face is cropped from YOLO detection
# Add 20% padding around detected region

# Before:
crop = frame[y1:y2, x1:x2]

# After:
pad = int((y2-y1) * 0.1)  # 10% padding
crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
```

### Fix 3: Add Embedding Validation
```python
# face_identity.py, _create_person() method

def _create_person(self, embedding, ...):
    if embedding is None:
        logger.error("Cannot create person: embedding is None!")
        return None
    
    serialized = self._serialize([embedding])
    if not serialized or len(serialized) < 100:  # Sanity check
        logger.error(f"Failed to serialize embedding!")
        return None
    
    # Continue...
```

### Fix 4: Reload Cache Periodically
```python
# face_identity.py, in __init__

def _refresh_known_embeddings_if_needed(self):
    now = time.monotonic()
    if now - self._last_refresh_at < 10:  # Change from 120 to 10 seconds
        return
    self._known = self._load_known_embeddings()
    self._last_refresh_at = now
```

---

## Complete Fix (If All Else Fails)

Create a validation wrapper:

```python
# Add to face_identity.py

class SafeFaceIdentityManager(FaceIdentityManager):
    """Wrapped version with extensive logging for debugging."""
    
    def identify(self, face_image: np.ndarray) -> FaceIdentityResult:
        # Log input
        h, w = face_image.shape[:2]
        logger.debug(f"[IDENTIFY] Input: {w}x{h} image")
        
        # Try embedding
        embedding = self._compute_embedding(face_image)
        logger.debug(f"[IDENTIFY] Embedding: {embedding is not None}")
        
        if embedding is None:
            logger.warning(f"[IDENTIFY] Failed: embedding is None for {w}x{h} image")
            return FaceIdentityResult(person_id="Unknown", ...)
        
        # Try matching
        match = self._find_best_match(embedding)
        logger.debug(f"[IDENTIFY] Match: {match is not None}")
        
        if match is not None:
            return FaceIdentityResult(person_id=match[0].person_id, ...)
        
        # Try creating
        logger.info(f"[IDENTIFY] Creating new person...")
        person_id = self._create_person(embedding)
        logger.info(f"[IDENTIFY] Created: {person_id}")
        
        return FaceIdentityResult(person_id=person_id, is_new=True)


# In main.py, change:
# identity_manager = FaceIdentityManager(db)
# To:
# identity_manager = SafeFaceIdentityManager(db)
```

---

## Expected Behavior (For Comparison)

When a **new user** is correctly stored:

1. **First sighting:**
   - Console: `Registered new face ID: PERSON_20260508_001 (initial pose: front)`
   - Database: New row in `persons` table with face_signature
   - Result: `is_new: true, confidence: 1.0`

2. **Second sighting (same person):**
   - Console: `COUNTED: PERSON_20260508_001 — count=1`
   - Result: `is_new: false, confidence: 0.65-0.85`

3. **After restart:**
   - On startup: `Face identity cache loaded: 5 persons, 12 total embeddings`
   - When person appears again: Recognized immediately (not as new)

If you're not seeing step 1, new users aren't being created.  
If you're seeing step 1 but not step 3, the database isn't persisting.

---

## What To Do Now

1. **Run:** `python debug_new_users.py` ← This identifies the exact problem
2. **Read** the test output carefully
3. **Apply** the fix corresponding to which test fails
4. **Test** with `python main.py` or the example script
5. **Share** the debug output if you need help

---

## Still Not Working?

Share the output of:
```bash
python debug_new_users.py  2>&1 > diagnostic_output.txt
# Then share the diagnostic_output.txt file
```

This will pinpoint exactly which part of the system is failing.
