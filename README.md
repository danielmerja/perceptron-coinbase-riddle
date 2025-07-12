# Perceptron Solution to the 100 Boxes Riddle

## The Riddle

Imagine 100 kids and 100 toy boxes, all shut at the start.
- Kid 1 opens every box.
- Kid 2 flips every 2nd box (closes if open, opens if closed).
- Kid 3 flips every 3rd box, and so on, up to Kid 100.

A box is flipped once for every kid whose number divides the box's number. Most boxes are flipped an even number of times (ending closed), but perfect squares are flipped an odd number of times (ending open).

**So, only the boxes with perfect square numbers (1, 4, 9, ..., 100) are open at the end.**

## Perceptron Approach

This script trains a simple perceptron (single neuron with sigmoid activation) to predict whether a box will be open (1) or closed (0) at the end, given its number (1–100).

- **Input:** Box number (normalized to 0–1)
- **Output:** 1 if open (perfect square), 0 if closed
- **Training:** Uses mean squared error and gradient descent

## Files
- `perceptron_boxes.py`: Main script for training and testing the perceptron
- `README.md`: This file

## How to Run

1. Make sure you have Python and NumPy installed:
   ```bash
   pip install numpy
   ```
2. Run the script:
   ```bash
   python perceptron_boxes.py
   ```

## Output
- The script prints the loss during training.
- After training, it prints the predicted and actual status (open/closed) for each box (1–100).
- It also prints the learned weight and bias.
