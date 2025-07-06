import torch


class WolframCA:
    """
    A PyTorch implementation of Wolfram's elementary cellular automata.
    """

    def __init__(self, rule_number, size=101, initial_condition="single", device="cpu"):
        """
        Initializes the cellular automaton.

        Args:
            rule_number (int): The Wolfram rule number (0-255).
            size (int): The number of cells in the automaton.
            initial_condition (str): The initial state type. Can be 'single' or 'random'.
            device (str): The torch device to run computations on ('cpu' or 'cuda').
        Raises:
            ValueError: If rule_number is not in [0, 255] or initial_condition is invalid.
        """
        if not 0 <= rule_number <= 255:
            raise ValueError("Rule number must be between 0 and 255.")
        self.rule_number = rule_number
        self.size = size
        self.device = device
        rule_bin = format(rule_number, "08b")
        self.rule = torch.tensor(
            [int(bit) for bit in rule_bin], dtype=torch.int8, device=self.device
        ).flip(0)
        if initial_condition == "single":
            self.state = torch.zeros(size, dtype=torch.int8, device=self.device)
            self.state[size // 2] = 1
        elif initial_condition == "random":
            self.state = torch.randint(
                0, 2, (size,), dtype=torch.int8, device=self.device
            )
        else:
            raise ValueError("initial_condition must be 'single' or 'random'.")
        self.conversion_weights = torch.tensor(
            [4, 2, 1], dtype=torch.int8, device=self.device
        )

    def step(self):
        """
        Computes the next state of the automaton using vectorized operations.
        Updates the state in-place.
        """
        left_neighbors = torch.roll(self.state, shifts=1, dims=0)
        right_neighbors = torch.roll(self.state, shifts=-1, dims=0)
        center = self.state
        neighborhoods = torch.stack([left_neighbors, center, right_neighbors])
        indices = torch.matmul(neighborhoods.T, self.conversion_weights).long()
        self.state = self.rule[indices]

    def run(self, generations):
        """
        Runs the simulation for a specified number of generations.

        Args:
            generations (int): Number of generations to simulate.
        Returns:
            torch.Tensor: A tensor containing the history of states (shape: generations x size).
        """
        history = torch.empty(
            (generations, self.size), dtype=torch.int8, device=self.device
        )
        for i in range(generations):
            history[i] = self.state
            self.step()
        return history
