import graph_generation
import os
import argparse
import sys


def write_command_to_file(log_file_path):

    # Get the command-line arguments
    command_args = " ".join(sys.argv)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a") as log_file:
        log_file.write(command_args + "\n")


def initialize_dir(compactness, n_segments):
    folder_name = "c" + str(compactness).replace(".", "_") + "__s" + str(n_segments)

    # Define the directory path
    dir_path = f"./data/processed/{folder_name}"

    # Check if the directory exists, and if not, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' was created.")
    write_command_to_file(dir_path + "/logfile.txt")
    return dir_path


def main(
    compactness, n_segments, file_id=None, downsample_factor: float = None
):  # TODO REMOVE downsampling arg
    dir_path = initialize_dir(compactness=compactness, n_segments=n_segments)

    if file_id:
        graph_generation.main(
            file_id=file_id,
            compactness=compactness,
            n_segments=n_segments,
            save_as=dir_path + f"/{file_id}.pt",
            downsample_factor=downsample_factor,  # TODO REMOVE downsampling arg
        )
    else:
        for i in range(1, 356):
            try:
                file_id = str(f"{i:03}")
                graph_generation.main(
                    file_id=file_id,
                    compactness=compactness,
                    n_segments=n_segments,
                    save_as=dir_path + f"/{file_id}.pt",
                    downsample_factor=downsample_factor,  # TODO REMOVE downsampling arg
                )
            except KeyboardInterrupt:
                break
            except KeyError as e:
                print(f"Ecountered error with file_id {file_id}\n\t{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main function with compactness and number of segments."
    )

    # Adding arguments
    parser.add_argument(
        "--compactness",
        type=float,
        required=True,
        help="The compactness parameter (float).",
    )
    parser.add_argument(
        "--n_segments",
        type=int,
        required=True,
        help="The number of segments (integer).",
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default=None,
        help="The number of segments (integer).",
    )

    parser.add_argument(
        "--downsample_factor",
        type=float,
        default=None,
        help="The number of segments (integer).",
    )

    # Parsing arguments
    args = parser.parse_args()

    main(args.compactness, args.n_segments, args.file_id, args.downsample_factor)
