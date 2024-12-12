# Data-Based Filtered Dissipation Rate Model for Multi-Modal Turbulent Combustion

This repository contains a deep neural network (DNN) model for use with the multi-modal manifold model by Mueller:

- M. E. Mueller, Physically-derived reduced-order manifold-based modeling for multi-modal turbulent combustion, Combustion and Flame 214 (2020) 287-305.

The DNN was trained using ```TensorFlow``` version 2.4.1 in ```Python 3```. The Anaconda environment containing all dependencies used to train and postprocess the DNN model is provided in ```tf-gpu.yml```. All associated libraries will automatically be installed to a new virtual environment named ```myenv``` via the command ```conda env create -n myenv -f tf-gpu.yml```.


## DNN Model Outputs

The DNN provides closure for the three filtered dissipation rates $$\chi_{ZZ}$$ (filtered mixture fraction dissipation rate), $$\chi_{Z \Lambda}$$ (filtered cross-dissipation rate), and $$\chi_{\Lambda \Lambda}$$ (filtered generalized progress variable dissipation rate), defined for general scalars $\psi$ and $\omega$ according to the general expression

$$ \chi_{\psi \omega} \equiv 2 D \nabla \psi \cdot \nabla \omega.$$

Note that $D$ is the molecular diffusivity, $\Lambda$ is the generalized progress variable, and $Z$ is the mixture fraction. To enforce physical constraints of positivity on $$\chi_{ZZ}$$ and $$\chi_{\Lambda \Lambda}$$ as well as the alignment of the mixture fraction and generalized progress variable gradients, the DNN provides closure by learning functional transformations of these three filtered dissipation rates. Namely, the DNN outputs comprise the natural logarithms of the filtered mixture fraction and generalized progress variable dissipation rates $ln\left(\widetilde{\chi}\_{Z Z}\right)$ and $ln\left(\widetilde{\chi}\_{\Lambda \Lambda}\right)$ as well as the inverse hyperbolic tangent of the pseudo-filtered alignment $tanh^{-1}\left(\widecheck{\Theta}\right)$, where $\widecheck{\Theta}\equiv\frac{\widetilde{\chi}_{Z\Lambda}}{\left(\widetilde{\chi}_{ZZ}\widetilde{\chi}_{\Lambda\Lambda}\right)^{1/2}}$. Details of the neural network architecture and training procedure are outlined in the following publication:

- C. E. Lacey, B. S. Soriano, M. Rieth, M. E. Mueller, J. H. Chen, Data-based filtered dissipation rate modeling for multi-modal turbulent combustion: Evaluating _a priori_ model generalizability, Combustion Theory and Modelling (2024) submitted.

We kindly ask that you cite the paper in any published work incorporating this DNN model.


## DNN Model Input Features

The DNN expects input features with corresponding column names as summarized in the following table:

| Input Feature | Description | Column Name   |
| :---:         |    :----   |        :---:   |
|   $\widetilde{\Lambda}$            | filtered progress variable                  |    ```FPROG```   |
|     $\Lambda_v$          | progress variable subfilter variance        |    ```PROGV```   |
|       $\lvert \nabla \widetilde \Lambda \rvert$        | magnitude of the filtered progress variable gradient                 |    ```dFPROG```  |
|        $\lvert \widetilde S \rvert$       | magnitude of the filtered strain rate        |    ```FS```      |
|        $\Delta_L \equiv V_{\rm stencil}^{1/3}$       | local filter size                 |    ```DELFILT``` |
|       $\widetilde{D}_{\Lambda}$        | filtered molecular diffusivity        |    ```FDIFF```   |
|       $\overline{\rho}$        | filtered density        |    ```FRHO```   |
|      $\overline{\dot{m}}_{\Lambda}$         | filtered progress variable source term                  |    ```FPROGSRC```|
|      $\widetilde \chi_{\Lambda \Lambda}$         | filtered progress variable dissipation rate       |    ```FCHIPP```  |
|      $\alpha$         | principal rate of strain (smallest)                 |    ```alpha```   |
|       $\beta$        | principal rate of strain (intermediate)        |    ```beta```    |
|      $\gamma$         | principal rate of strain (largest)                  |    ```gamma```   |
|        $e_{\alpha}\cdot \nabla \widetilde{\Lambda}/\lvert \nabla \widetilde{\Lambda}\rvert$       | alignment of (smallest) principal rate of strain with progress variable gradient        |    ```alpha_align```   |
|       $e_{\beta}\cdot \nabla \widetilde{\Lambda}/\lvert \nabla \widetilde{\Lambda}\rvert$        | alignment of (intermediate) principal rate of strain with progress variable gradient                  |    ```beta_align```    |
|       $e_{\gamma}\cdot \nabla \widetilde{\Lambda}/\lvert \nabla \widetilde{\Lambda}\rvert$        | alignment of (largest) principal rate of strain with progress variable gradient        |    ```gamma_align```   |

**Note:** This particular DNN expects a ***normalized mixture fraction*** and ***normalized progress variable***. Before generating DNN predictions, ensure that all the input features incorporating the mixture fraction or generalized progress variable are first normalized appropriately by the maximum values in your domain of interest -- that is, $Z_{max}$ and $\Lambda_{max}$, respectively. Finally, the filtered reference species source term $\overline{\dot{m}}\_{\rm R}$ corresponds to the reference species $Y_{\rm R} \equiv Y_{\rm H_2} + Y_{\rm H_2 O} + Y_{\rm CO} + Y_{\rm CO_2}$, possessing dimensions of $ML^{-3}T^{-1}$.

## Generating DNN Model Predictions

An example script that loads the DNN model and generates model predictions for a test dataset is provided below. The script assumes the DNN model is stored in the directory ```filt_multi_modal_dissrate_dnn/``` and the test dataset is stored in the working directory as a CSV file named ```test_data.csv```.

```python
# Import required libraries
import pandas as pd
import tensorflow as tf

# Define directories and column names
test_data_path = 'test_data.csv'
dnn_path = 'filt_multi_modal_dissrate_dnn'
label_names =  ['g_' + str(i) for i in range(32)]
feature_names = ['FPROG', \
                 'PROGV', \
                 'dFPROG', \
                 'FS', \
                 'DELFILT', \
                 'FDIFF', \
                 'FPROGSRC', \
                 'FRHO', \
                 'FCHIPP', \
                 'alpha', \
                 'beta', \
                 'gamma', \
                 'alpha_align', \
                 'beta_align', \
                 'gamma_align']

# Load testing dataset (example for CSV data)
test_df = pd.read_csv(test_data_path)
test_features = test_df[feature_names]

# Load DNN model
dnn_model = tf.keras.models.load_model(dnn_path)

# Generate test predictions
dnn_predictions = pd.DataFrame(dnn_model.predict(test_features), columns=label_names)

```