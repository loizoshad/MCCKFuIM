#! /usr/bin/env python3

import numpy as np


def main():
    # Read data from the csv file
    data = np.genfromtxt('hover_inputs_data2.csv', delimiter=',')

    # Remove the first row which contains the labels
    data = data[1:]

    # Extract the columns
    u1 = data[:, 0]; u2 = data[:, 1]; u3 = data[:, 2]; u4 = data[:, 3]

    # Compute the average value of each column
    u1_avg = np.mean(u1); u2_avg = np.mean(u2); u3_avg = np.mean(u3); u4_avg = np.mean(u4)

    # Compute the standard deviation of each column
    u1_std = np.std(u1); u2_std = np.std(u2); u3_std = np.std(u3); u4_std = np.std(u4)

    # Print the results
    print(f'Average values:')
    print(f'U1: {u1_avg:.8f} | U2: {u2_avg:.8f} | U3: {u3_avg:.8f} | U4: {u4_avg:.8f}')
    print(f'Standard deviations:')
    print(f'U1: {u1_std:.8f} | U2: {u2_std:.8f} | U3: {u3_std:.8f} | U4: {u4_std:.8f}')

if __name__ == '__main__':
    main()















