from network.reservoir import RCNetwork

reservoir = RCNetwork(
    dim_reservoir=1000,
    dim_in=3,
    dim_out=3,
    alpha=0.1,
    rho=1.5,
    sigma_rec=0.1
)

reservoir.train_dynamic_target(5, 5, do_plot=True, seed=10)