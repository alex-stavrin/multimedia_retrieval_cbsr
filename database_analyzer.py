import pandas as pd
import matplotlib.pyplot as plt


def show_histogram_from_csv(
    csv_path: str, n_bins: int, header_property: str = ""
):
    if not csv_path.endswith(".csv"):
        pass

    df = pd.read_csv(csv_path)
    df[header_property].hist(bins=n_bins)

    plt.show()


def main():
    # Only import when using main, to avoid lib cluttering
    import argparse

    parser = argparse.ArgumentParser(description="Analyze CSV database.")

    parser.add_argument(
        "-i", "--input", type=str, help="Path to the CSV file to analyze"
    )
    parser.add_argument(
        "-p",
        "--property",
        type=str,
        help="Header property wanted from CSV file",
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=20,
        help="Number of bins for a histogram"
    )

    args = parser.parse_args()

    show_histogram_from_csv(args.input, args.bins, args.property)


if __name__ == "__main__":
    main()
