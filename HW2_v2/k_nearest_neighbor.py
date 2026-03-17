import torch
import numpy as np

class KNearestNeighbor(object):
    """ 
        A kNN classifier with L2 distance using PyTorch 
    """

    # initialize torch variables
    def __init__(self):
        pass

    def train(self, X, y):

        """

        Train the classifier. For k-nearest neighbors this is just memorizing the training data.

        Inputs:
            X: PyTorch Tensor of shape (num_train, D)
            y: PyTorch Tensor of shape (num_train,)
        
        """

        self.X_train = X
        self.y_train = y


    def predict(self, X, k=1, num_loops=0):
    
        """
            Predict labels for test data using this classifier.

            Inputs:

            - X: A torch tensor of shape (num_test, D) containing test data consisting
                of num_test samples each of dimension D.
            - k: The number of nearest neighbors that vote for the predicted labels.
            - num_loops: Determines which implementation to use to compute distances
            between training points and testing points.

            Returns:
            - y: A torch tensor of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].  
        
        """

        if num_loops == 0:

            dists = self.compute_distances_no_loops(X)

        elif num_loops == 1:

            dists = self.compute_distances_one_loop(X)

        elif num_loops == 2:

            dists = self.compute_distances_two_loops(X)

        else:

            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    
    def compute_distances_two_loops(self, X):

        """
            Compute the distance between each test point in X and each training point
            in self.X_train using a nested loop over both the training data and the 
            test data.

            Inputs:
            - X: A numpy array of shape (num_test, D) containing test data.

            Returns:
            - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            is the Euclidean distance between the ith test point and the jth training
            point.
        """

        num_test = X.shape[0]

        num_train = self.X_train.shape[0]

        dists = torch.zeros((num_test, num_train), device=X.device)

        for i in range(num_test):
            for j in range(num_train):
                
                #####################################################################
                # TODO Task 2.1:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                # dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
                dists[i, j] = torch.norm(X[i] - self.X_train[j])
                ## (i, j) is the distance between the i-th test point and the j-th training point
                # torch norm with p=2 (default) is the Euclidean distance between two vectors
                #####################################################################s
                #                       END OF YOUR CODE                            #
                #####################################################################

        return dists


    def compute_distances_one_loop(self, X):

        """
            Compute the distance between each test point in X and each training point
            in self.X_train using a single loop over the test data.

            Input / Output: Same as compute_distances_two_loops
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dists = torch.zeros( (num_test, num_train) , device=X.device)

        for i in range(num_test):

            #######################################################################
            # TODO  Task 2.4:                                                        #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].  #broadcasting         #
            #######################################################################
            # X[i] is the i-th test point
            # computing the difference between the i-th test point and all training points
            # dimension of X[i] is (1, D); dimension of self.X_train is (num_train, D)
            # resulting dimension is (num_train, D)
            diff = X[i] - self.X_train
            # calculating the squared distance between the i-th test point and all training points
            dists[i, :] = torch.sqrt(torch.sum(diff ** 2, dim=1))
            #######################################################################
            #                       END OF YOUR CODE                                #
            #######################################################################

        return dists


    def compute_distances_no_loops(self, X):
        
        """
            Compute the distance between each test point in X and each training point
            in self.X_train using no explicit loops.

            Input / Output: Same as compute_distances_two_loops
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        #########################################################################
        # TODO Task 2.5:                                                        #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: The naive implementation uses extensive amount of memory. The   #
        #       solution is to use equation (x - y)^2 = x^2 - 2xy + y^2. Try to #
        #       formulate it using matrix multiplication and two broadcast sums.#
        #########################################################################

        # using the frobenius distance, we can simplify the Eucledian distance (x - y) ^ 2)
        # using the formula (x - y) ^ 2 = x^2 - 2xy + y^2
        # in matrix form, it becomes: X^2 + y^2 - 2 * dot product (X, Y.T)
        
        # X^2, keeping dimension to be (num_test, 1)
        test_squared = torch.sum(X ** 2, dim=1, keepdim=True) 
        # Y^2, transposing the dimension to be (1, num_train)
        train_squared = torch.sum(self.X_train ** 2, dim=1, keepdim=True).T
        # dot product of X and Y.T, dimension is (num_test, num_train)
        dot_product = torch.matmul(X, self.X_train.T)
        dists = torch.sqrt(test_squared + train_squared - 2 * dot_product)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################

        return dists

    def predict_labels(self, dists, k=1):

        """
            Given a matrix of distances between test points and training points,
            predict a label for each test point.

            Inputs:
            - dists: A torch tensor of shape (num_test, num_train) where dists[i, j]
            gives the distance betwen the ith test point and the jth training point.

            Returns:
            - y: A torch tensor of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].  
        """

        num_test = dists.shape[0]

        # Create y_pred on the same device as dists (CPU or GPU)
        y_pred = torch.zeros( num_test, dtype=torch.long, device=dists.device )
        
        for i in range(num_test):

            #########################################################################
            # TODO  Task 2.3:                                                       #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort or torch.topk.               #
            #########################################################################
            # closest_indices = np.argsort(dists[i])[:k]

            # sorting the distances in ascending order and taking the first k indices
            closest_indices = torch.argsort(dists[i])[:k]  
            # getting the labels of the k nearest neighbors
            closest_y = self.y_train[closest_indices]
            # finding the most common label in the list closest_y of labels
            y_pred[i] = torch.bincount(closest_y).argmax()
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label. Hint: Look up the functions torch.bincount and torch.argmax.   #
            #########################################################################
            
            # Find the most common label
            # torch.bincount works like np.bincount
            pass

        #########################################################################
        #                           END OF YOUR CODE                            # 
        #########################################################################

        return y_pred