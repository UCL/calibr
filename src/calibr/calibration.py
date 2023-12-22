import asyncio

import numpy as np


def calibrate(log_target_density, batch_size, num_initial_points, num_iterations, seed):
    rng = np.random.default_rng(seed)
    inputs, outputs = [], []
    for iteration in range(num_iterations):
        if iteration == 0:
            new_inputs = initial_design(rng, num_initial_points)
            emulator = Emulator()
        else:
            new_inputs = acquire_next_batch(rng, batch_size, emulator)
        new_outputs = asyncio.run(
            asyncio.gather(*(log_target_density(input_) for input_ in new_inputs))
        )
        inputs.append(new_inputs)
        outputs.append(new_outputs)
        emulator.fit(new_inputs, new_outputs)
        if converged(emulator):
            break
    return emulator, inputs, outputs
