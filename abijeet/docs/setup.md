# prerequisites (one-time)
# ../venv312/bin/pip install -r requirements.txt

# run to remove the preexisting database and create a new one with the current schema
rm attendance.db attendance.db-shm attendance.db-wal
../venv312/bin/python setup.py


# to run the project use this command
../venv312/bin/python main.py
../venv312/bin/python frontend/server.py



# clean everything
rm -rf logs/* .cache __pycache__ frontend/__pycache__
rm -f attendance.db attendance.db-shm attendance.db-wal
../venv312/bin/python setup.py
