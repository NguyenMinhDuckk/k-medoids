import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_cluster_means(csv_file='clustering_results.csv', cluster_column='cluster'):
    """
    Calculate mean values of specified columns for each cluster.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing clustered data (default: 'clustering_results.csv')
    cluster_column : str
        Name of the column that contains cluster assignments
        
    Returns:
    --------
    DataFrame with mean values for each cluster
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Columns to analyze
    columns_of_interest = [
        'Depression', 
        'Have you ever had suicidal thoughts ?=No',
        'Age',
        'Gender=Female',
        'Academic Pressure', 
        'Study Satisfaction', 
        'Sleep Duration', 
        'Dietary Habits', 
        'Work/Study Hours'
    ]
    
    # Verify all columns exist in the dataframe
    missing_columns = [col for col in columns_of_interest if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing from the dataset: {missing_columns}")
        # Filter out missing columns
        columns_of_interest = [col for col in columns_of_interest if col in df.columns]
    
    # Check if cluster column exists
    if cluster_column not in df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in the dataset.")
    
    # Check if Depression column exists
    if 'Depression' not in df.columns:
        raise ValueError("'Depression' column not found in the dataset.")
    
    # Calculate mean for each column by cluster
    cluster_means = df.groupby(cluster_column)[columns_of_interest].mean()
    
    # Calculate Depression counts by cluster
    depression_counts = df[df['Depression'] == 1].groupby(cluster_column).size()
    non_depression_counts = df[df['Depression'] == 0].groupby(cluster_column).size()
    
    # Add count columns to the cluster_means DataFrame
    cluster_means['Depression_count'] = depression_counts
    cluster_means['Non_Depression_count'] = non_depression_counts
    
    # Handle any missing counts (clusters that might not have any Depression=1 or Depression=0)
    cluster_means['Depression_count'] = cluster_means['Depression_count'].fillna(0).astype(int)
    cluster_means['Non_Depression_count'] = cluster_means['Non_Depression_count'].fillna(0).astype(int)
    
    return cluster_means

def create_visualization(cluster_means, output_image='cluster_results.png'):
    """
    Create and save a table visualization of the cluster means with proper text wrapping.
    
    Parameters:
    -----------
    cluster_means : DataFrame
        DataFrame containing mean values by cluster
    output_image : str
        Path to save the visualization image (default: 'cluster_results.png')
    """
    # Reset index to make cluster column visible in the table
    table_data = cluster_means.reset_index()
    
    # Round float values for better display
    for col in table_data.columns:
        if col != 'cluster' and table_data[col].dtype == 'float64':
            table_data[col] = table_data[col].round(2)
    
    # Function to wrap text
    def wrap_text(text, width=12):
        """Wrap text to fit within a cell"""
        if len(str(text)) <= width:
            return str(text)
        
        # Handle special column names
        if str(text) == 'Depression_count':
            return 'Depression\nCount'
        if str(text) == 'Non_Depression_count':
            return 'Non-Depression\nCount'
        
        # For longer text, split by words
        words = str(text).split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + len(current_line) <= width:
                current_line.append(word)
                current_length += len(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)
    
    # Get number of rows and columns
    n_rows, n_cols = table_data.shape
    
    # Prepare wrapped headers
    wrapped_headers = [wrap_text(col, width=15) for col in table_data.columns]
    
    # Calculate header heights (number of lines)
    header_heights = [len(header.split('\n')) for header in wrapped_headers]
    max_header_height = max(header_heights)
    header_height = max(1, max_header_height) * 0.5  # Increased height for header
    
    # Create figure with fixed cell size and adjust for header height
    cell_width, cell_height = 1.2, 0.6
    fig_width = n_cols * cell_width + 1  # Add some padding
    # No extra space between header and data rows
    fig_height = (n_rows * cell_height) + header_height + 0.5  # Reduced padding
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    
    # Hide axes
    ax.axis('off')
    
    # Define all horizontal line positions
    y_positions = []
    
    # Top line (figure top)
    y_positions.append(fig_height - 0.25)  # Slight offset from top
    
    # Line below header
    y_positions.append(fig_height - header_height - 0.25)
    
    # Lines for data rows
    for i in range(1, n_rows + 1):
        y_positions.append(fig_height - header_height - i * cell_height - 0.25)
    
    # Draw all horizontal lines
    for y_pos in y_positions:
        ax.axhline(y=y_pos, color='black', linewidth=1.5)
    
    # Draw vertical lines
    for j in range(n_cols + 1):
        x_pos = j * cell_width
        ax.axvline(x=x_pos, color='black', linewidth=1.5)
    
    # Add column headers
    header_y = fig_height - header_height/2 - 0.25
    for j, header in enumerate(wrapped_headers):
        ax.text(j*cell_width + cell_width/2, header_y, 
                header, ha='center', va='center', fontsize=9, fontweight='bold', 
                wrap=True, multialignment='center')
    
    # Add data rows (directly below header with no gap)
    for i in range(n_rows):
        for j in range(n_cols):
            value = str(table_data.iloc[i, j])
            # Position: center of cell
            data_y = fig_height - header_height - (i + 0.5) * cell_height - 0.25
            ax.text(j*cell_width + cell_width/2, data_y, 
                    value, ha='center', va='center', fontsize=10)
    
    # Add title
    plt.title('Mean Values by Cluster', fontsize=16, pad=20)
    
    # Set proper limits
    ax.set_xlim(0, n_cols * cell_width)
    ax.set_ylim(0, fig_height)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Table visualization saved to {output_image}")
    
    return plt

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate mean values by cluster for specified columns.')
    parser.add_argument('--input_file', type=str, default='clustering_results.csv', 
                        help='Path to the input CSV file (default: clustering_results.csv)')
    parser.add_argument('--cluster_col', type=str, default='cluster', 
                        help='Column name containing cluster assignments (default: "cluster")')
    parser.add_argument('--output_file', type=str, help='Path to save the results CSV (optional)')
    parser.add_argument('--output_image', type=str, default='cluster_results.png',
                        help='Path to save the visualization image (default: cluster_results.png)')
    
    args = parser.parse_args()
    
    # Calculate means
    means_df = calculate_cluster_means(args.input_file, args.cluster_col)
    
    # Display results
    print("\nMean values by cluster:")
    print(means_df)
    
    # Save to CSV file if specified
    if args.output_file:
        means_df.to_csv(args.output_file)
        print(f"Results saved to {args.output_file}")
    
    # Create and save visualization
    create_visualization(means_df, args.output_image)

if __name__ == "__main__":
    main()
