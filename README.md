# VAEP-Thesis
The repository for my senior honors thesis from Duke University

## Abstract
This thesis seeks to enhance the analytical modeling of player actions in soccer utilizing Division 1 women’s collegiate soccer event-level data. By focusing on the Valuing Actions by Estimating Probabilities (VAEP) framework, this research addresses the need for model updates in light of a new and more detailed data version available to D1 women’s teams and dives deeper into understanding the components that render a player's actions valuable. Employing data accessed through an API from the Duke team's provider, Wyscout, this study intricately analyzes the transition from Version 2 (V2) to Version 3 (V3) of the Wyscout event data, adapting it to the Soccer Player Action Description Language (SPADL) for model compatibility and analysis. Additionally, through evaluating the performance of the VAEP model and completing a variable importance analysis on it, the research provides a detailed comparison between model versions and the significance of various actions within the game. Findings highlight the model's enhancements and the critical factors contributing to player performance, offering a useful tool for coaches to refine their in-game tactics and player recruitment plans. Moreover, this thesis outlines potential future directions, including utilizing this same structure to analyze players and leagues around the world, to augment the applicability of VAEP across broader soccer contexts. Overall, this study aims to contribute to soccer analytics by elucidating key aspects that influence positive game outcomes, thus offering a replicable foundation for further research in this field.

### File Directory
- main.py: Contains the majority of functions created and code tested I used in writing my thesis
- data_transformations.py: Contains various data functions needed to update and run my code
- socceraction/: contains a version of the socceraction package (github.com/ML-KULeuven/socceraction) modified to work with Wyscout V3 data; Most changes I needed to make and my visualization code can be found in socceraction/vaep/base.py
- Thesis Paper: Current draft of my thesis paper
- Thesis Presentation: PDF version of the presentation I made to a committee consisting of Duke Statistics Department faculty where I defended my thesis
