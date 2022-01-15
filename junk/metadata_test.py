"""script to test serialization of experiment meta-data"""
from datetime import datetime
from git import Repo

if __name__ == '__main__':

    # Get git hash
    repo = Repo(search_parent_directories=True)
    print("git commit:")
    print(repo.active_branch.commit)

    # Get timestamp
    t = datetime.utcnow()
    print("timestamp:")
    print(t.isoformat())