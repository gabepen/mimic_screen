import pandas as pd
import argparse

def search_pca_results(csv_file, x_min, x_max, y_min, y_max):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Filter the DataFrame for the given coordinate range
    filtered_df = df[(df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)]
    
    # Output the last two values of the matching rows
    result = filtered_df.iloc[:, -2:]
    return result

def main():
    parser = argparse.ArgumentParser(description='Search PCA results within a coordinate range.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing PCA results.')
    parser.add_argument('x_min', type=float, help='Minimum x coordinate.')
    parser.add_argument('x_max', type=float, help='Maximum x coordinate.')
    parser.add_argument('y_min', type=float, help='Minimum y coordinate.')
    parser.add_argument('y_max', type=float, help='Maximum y coordinate.')
    
    args = parser.parse_args()
    
    matching_rows = search_pca_results(args.csv_file, args.x_min, args.x_max, args.y_min, args.y_max)
    print(matching_rows)

if __name__ == '__main__':
    main()