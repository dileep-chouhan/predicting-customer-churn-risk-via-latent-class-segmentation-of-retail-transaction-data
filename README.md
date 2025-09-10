# Predicting Customer Churn Risk via Latent Class Segmentation of Retail Transaction Data

## Overview

This project aims to identify high-risk customer segments susceptible to churn within a retail business using latent class segmentation techniques.  The analysis leverages transactional data to uncover underlying patterns in customer behavior that predict churn.  This information can then be used to inform targeted retention strategies and ultimately improve customer lifetime value.  The project performs data preprocessing, latent class model fitting, and visualization of key findings.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed.  Then, install the required Python libraries listed above using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

## Example Output

The script will print key findings to the console, including details about the fitted latent class model (e.g., number of segments, log-likelihood, BIC).  Additionally, the script will generate several visualization files (e.g., plots showing the distribution of key variables across customer segments, churn probabilities by segment).  These plots will be saved in the `output` directory.  Example output files include:

* `segment_characteristics.png`:  Illustrates key characteristics of each identified customer segment.
* `churn_probability_by_segment.png`: Shows the churn probability for each identified segment.


## Data

The project requires a dataset containing transactional data.  A sample dataset is provided in the `data` directory.  The specific format and required columns are detailed within the `main.py` script.  You may need to adapt the data loading section to your specific data format.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[Specify your license here, e.g., MIT License]