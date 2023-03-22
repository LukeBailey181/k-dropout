import os
import subprocess
import hashlib
import wandb


def get_short_hash(filename, length=6):
    # Read the contents of the file into a bytes object
    with open(filename, "rb") as f:
        file_data = f.read()

    # Calculate the hash value of the file contents
    hash_value = hashlib.sha256(file_data).hexdigest()

    # Convert the hash value to a 4-character hexadecimal string
    short_hash = hash_value[:length]

    return short_hash


def write_git_snapshot():
    # Get the latest git commit
    git_commit = subprocess.check_output(["git", "log", "-1"]).decode("utf-8").strip()
    git_commit_id = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )

    # Get the diff and untracked files
    git_diff = subprocess.check_output(["git", "diff"]).decode("utf-8")
    git_untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"]
    ).decode("utf-8")

    # Create the artifacts directory if it doesn't exist
    if not os.path.exists("artifacts"):
        os.mkdir("artifacts")

    # Create a unique and helpful filename for the git snapshot
    temp_file = "git_snapshot_temp"

    # Write the git snapshot to a file in markdown format
    with open(temp_file, "w") as f:
        f.write("## Git Snapshot\n\n")
        f.write(f"Latest commit: `{git_commit}`\n\n")
        f.write("### Git diff:\n\n```\n")
        f.write(git_diff)
        f.write("```\n\n")
        f.write("### Untracked files:\n\n")

        # Iterate over the untracked files and append their contents to the snapshot file in markdown format
        for file in git_untracked.split("\n"):
            if file:
                f.write(f"#### {file}\n\n```\n")
                with open(file, "r") as uf:
                    f.write(uf.read())
                f.write("```\n\n")

    # Append the short hash of the git snapshot file to the filename for easy identification
    short_hash = get_short_hash(temp_file)
    filename = f"git_snapshot_{git_commit_id[:6]}_{short_hash}"
    filepath = os.path.join("artifacts", f"{filename}.md")
    os.rename(temp_file, filepath)

    # Return the path to the git snapshot file
    return filename, filepath


if __name__ == "__main__":
    _, filepath = write_git_snapshot()
    print(f"Git snapshot saved to {filepath}")
