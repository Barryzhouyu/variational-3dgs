import csv
import argparse

def calculate_column_averages(file_path):
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Initialize a list to store sum of each column and a counter for rows
        sums = None
        row_count = 0
        
        for row in csvreader:
            # Skip header if present

            values = [float(value) for value in row[2:]]  # Skip the first two columns
            if sums is None:
                sums = values
            else:
                sums = [s + v for s, v in zip(sums, values)]
            
            row_count += 1
        
        # Calculate averages
        averages = [s / row_count for s in sums]
        
        return row_count, averages

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='dataset name')
    
    args = parser.parse_args()
    # Path to your CSV file
    if args.dataset_name == "LF": 
        file_path = './output/eval_results_LF.csv'

        # Calculate and print the averages
        row_count, averages = calculate_column_averages(file_path)
        results = f"LF datset total {row_count} Scenes, Averaged Results: PSNR {averages[0]} SSIM {averages[1]} LPIPS {averages[2]} AUSE {averages[3]} NLL {averages[4]} Depth AUSE {averages[5]}"
        print(results)
    elif args.dataset_name == "LLFF": 
        file_path = './output/eval_results_LLFF.csv'

        # Calculate and print the averages
        row_count, averages = calculate_column_averages(file_path)
        results = f"LLFF dataset total {row_count} Scenes, Averaged Results: PSNR {averages[0]} SSIM {averages[1]} LPIPS {averages[2]} AUSE {averages[3]} NLL {averages[4]}"
        print(results)
    else: 
        raise ValueError("Datset name is required. ")