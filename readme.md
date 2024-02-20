# Soft-constrained Schr&ouml;dinger Bridge: a Stochastic Control Approach

This repo contains the implementation  of the paper Soft-constrained Schr&ouml;dinger Bridge: a Stochastic Control Approach for the MNIST dataset

## Dependencies

* PyTorch

* PyYAML

* tqdm

* PILLOW

## Running Experiments

Configuration file is stored as 'config.yml'.

### Training
### Density Ratio Estimation

To  start the training of the logarithm of density ratios between the reference dataset and the objective dataset, run the command

```bash
python3 density_ratio.py
```
A new directory named 'Model' will be generated, and the trained logarithm density ratio will be saved as "checkpoint\_density\_estimation.pth" inside Models.

### Score Estimation

To start the training of the score functions with a particular value of $\beta$, run the following command, replacing value with the desired $\beta$ value (e.g., $\beta$ = 2)  

```bash
python3 score.py --beta value
```

For instance, to train the score functions for $\beta$ = 2.0, simply execute

```bash
python3 score.py --beta 2.0
```

This will train two score functions using the respective loss functions

$$
    E_{x \sim  \mu_{obj}} \left[E_{\tilde{x} \sim \mathcal{N}(x, \tilde{\sigma}^2 \mathbf{I})} 
    \left[\left\| s_\theta( \tilde{x}, \tilde{\sigma}) + \frac{\tilde{x} - x}{\tilde{\sigma}^2 } \right\|^2  \right].  \left( \frac{f_{ref}(x)}{f_{obj}(x)} \right)^{\frac{1}{1 + \beta}}  \right]. 
$$

and 

$$
    E_{x \sim  \mu_{ref}} \left[E_{\tilde{x} \sim \mathcal{N}(x, \tilde{\sigma}^2 \mathbf{I})} 
    \left[\left\| s_\theta( \tilde{x}, \tilde{\sigma}) + \frac{\tilde{x}- x}{\tilde{\sigma}^2 } \right\|^2  \right].  \left( \frac{f_{obj}(x)}{f_{ref}(x)} \right)^{\frac{\beta}{1 + \beta}}  \right]. 
$$

These two trained score functions will be saved in "/Model/score_obj" and "/Model/score_ref" respectively. The saved checkpoint file will be named 'checkpoint.pth' in each directory. The number of iterations needed to train the score functions may vary depending on the value of the parameter $\beta$. You may change it in the configuration file.

To only train the first score function (with respect to the objective dataset) with a particular value of $\beta$, execute

```bash
python3 score_obj.py --beta value
```
with value replaced by the desired value of $\beta$

To only train the second score function (with respect to the reference dataset) with a particular value of $\beta$, execute

```bash
python3 score_ref.py --beta value
```
with value replaced by the desired value of $\beta$




### Sampling

We can produce samples to folder `Samples`  for any value of $\beta$ by running the below command with the value replaced by the desired value of $\beta$ 

```bash
python3 runner.py --beta value
```

### Training and Sampling

To train the models and generate samples together for a specific $\beta$ value, use the following command, replacing "value" with your desired $\beta$ value.

```bash
python3 main.py --beta value
```


## References

Large parts of the code are derived from [this Github repo](https://github.com/ermongroup/ncsn)  and [this Github repo](https://github.com/YangLabHKUST/DGLSB)



