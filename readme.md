# comp7502-project

Image Processing - Deck Recognition

The project aims to produce a program to recognize decks of card one by one. The input image will be restricted to the following limitation to maximize the efficiency of the program.

1) The image will only contain one single playing card.
2) The concerned playing card might be covered partially
3) The concerned playing card might be taken at an angle (prespective projection effect)
4) The concerned playing card might be bent at an angle (prespective projection effect + bent effect)

Owning to the above restriction, the program will be divided into the following scenario:

Scenario 1: upright with no cover
Scenario 2: upright partially covered
Scenario 3: at a angle with no cover
Scenario 4: at a angle partially covered
Scenario 5: bent with no cover
Scenario 6: bent partially covered

Precomputations

1) Collect training images - take photos and use the affine transform to rotate them to an upright position for matching
 + Plus collect the lighting masks (by taking a photo of a piece of white paper) to correct for lighting

Step 1: Card recognition

Input: a photo with one single playing card either covered or without cover
1) Apply grayscaling
2) Find contours / edges / corners of the cards 
3) Affine Transform
Output: a normalized and straightened playing card 

Step 2: Rank recognition

Input: a normalized and straightened playing card 
1) Feature correlation
2) Machine-learning based identification
Output: rank

Step 3: Suit recognition

Input: a normalized and straightened playing card 
1) Feature correlation
2) Machine-learning based identification
Output: suit
