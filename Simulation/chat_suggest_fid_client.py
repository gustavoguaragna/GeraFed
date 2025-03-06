class FlowerClient(NumPyClient):
    def __init__(self, cid, net_alvo, net_gen, dataset, lr_alvo, lr_gen, latent_dim, context, agg, model):
        self.cid = cid
        self.net_alvo = net_alvo
        self.net_gen = net_gen
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_alvo.to(self.device)
        self.net_gen.to(self.device)
        self.dataset = dataset
        self.lr_alvo = lr_alvo
        self.lr_gen = lr_gen
        self.latent_dim = latent_dim
        self.client_state = context.state
        self.agg = agg
        self.model = model
        # Don't load data in __init__, will load in fit() after FID calculation
        self.trainloader = None
        self.valloader = None

    def calculate_client_fid(self):
        # Similar to server's FID calculation but for client's model
        # Return FID scores per class
        # ... (FID calculation code) ...
        return client_fid_scores

    def filter_and_load_data(self, partition_id, num_partitions, niid, alpha_dir, batch_size, server_fid_scores):
        # Calculate client's FID scores
        client_fid_scores = self.calculate_client_fid()
        
        # Compare FID scores and determine which classes to train on
        classes_to_train = []
        for class_idx in range(10):  # Assuming 10 classes
            if client_fid_scores[class_idx] > server_fid_scores[class_idx]:
                classes_to_train.append(class_idx)
        
        # Load data with filtered classes
        trainloader, valloader = load_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            niid=niid,
            alpha_dir=alpha_dir,
            batch_size=batch_size,
            filter_classes=classes_to_train  # New parameter
        )
        
        return trainloader, valloader

    def fit(self, parameters, config):
        # Load and filter data based on FID scores from server
        if self.trainloader is None and "fid_scores" in config:
            self.trainloader, self.valloader = self.filter_and_load_data(
                partition_id=self.cid,
                num_partitions=config["num_partitions"],
                niid=config["niid"],
                alpha_dir=config["alpha_dir"],
                batch_size=config["batch_size"],
                server_fid_scores=config["fid_scores"]
            )
        
        # Rest of the fit method...

 def load_data(partition_id, num_partitions, niid, alpha_dir, batch_size, filter_classes=None):
        """Load MNIST with filtered classes based on FID comparison."""
        global fds

        if fds is None:
            # ... existing partitioner code ...
            fds = FederatedDataset(
                dataset="mnist",
                partitioners={"train": partitioner}
            )

        train_partition = fds.load_partition(partition_id, split="train")
        test_partition = fds.load_split("test")

        if filter_classes is not None:
            # Filter data to only include specified classes
            train_partition = train_partition.filter(
                lambda x: x["label"] in filter_classes
            )

        # ... rest of the function (transforms, etc.) ...
        return trainloader, testloader