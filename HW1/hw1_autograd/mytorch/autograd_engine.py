import numpy as np
from mytorch.utils import GradientBuffer


class Operation:
    def __init__(self, inputs, output, gradients_to_update, backward_operation):
        """
        Args:
            - inputs: operation inputs (numpy.ndarray)
            - outputs: operation output (numpy.ndarray)
            - gradients_to_update: parameter gradients if for parameter of ,
                        network or None (numpy.ndarray, None)
            - backward_operation: backward function for nn/functional.py.
                        When passing a function you don't need inputs or parentheses.
        Note: You do not need to modify anything here
        """
        self.inputs = inputs
        self.output = output
        self.gradients_to_update = gradients_to_update
        self.backward_operation = backward_operation


class Autograd:
    def __init__(self):
        """
        WARNING: DO NOT MODIFY THIS METHOD!
        A check to make sure you don't create more than 1 Autograd at a time. You can remove
        this if you want to do multiple in parallel. We do not recommend this
        """
        if getattr(self.__class__, "_has_instance", False):
            raise RuntimeError("Cannot create more than 1 Autograd instance")
        self.__class__._has_instance = True

        self.gradient_buffer = GradientBuffer()
        self.operation_list = []

    def __del__(self):
        """
        WARNING: DO NOT MODIFY THIS METHOD!
        Class destructor. We use this for testing purposes.
        """
        del self.gradient_buffer
        del self.operation_list
        self.__class__._has_instance = False

    def add_operation(self, inputs, output, gradients_to_update, backward_operation):
        """
        Adds operation to operation list and puts gradients in gradient buffer for tracking
        Args:
            - inputs: operation inputs (numpy.ndarray)
            - outputs: operation output (numpy.ndarray)
            - gradients_to_update: parameter gradients if for parameter of
                        network or None (numpy.ndarray, None)
                NOTE: Given the linear layer as shown in the writeup section
                    2.4 there are 2 kinds of inputs to an operation:
                    1) one that requires gradients to be internally tracked
                        ex. input (X) to a layer
                    2) one that requires gradient to be externally tracked
                        ex. weight matrix (W) of a layer (so we can track dW)
            - backward_operation: backward function for nn/functional.py.
                        When passing a function you don't need inputs or parentheses.
        Returns:
            No return required
        """
        if len(inputs) != len(gradients_to_update):
            raise Exception(
                "Number of inputs must match the number of gradients to update!"
            )
        for input, gradient in zip(inputs, gradients_to_update):
            if gradient is not None:
                self.gradient_buffer.add_spot(input)


        # Append an Operation object to the self.operation_list
        operation = Operation(inputs, output, gradients_to_update, backward_operation)
        self.operation_list.append(operation)

    def backward(self, divergence):       
        #  Iterate through the operation list in reverse order.
        for operation in reversed(self.operation_list):
            # Step 2: Determine the gradient to be propagated for this operation.
            grad_of_output = divergence if operation is self.operation_list[-1] else self.gradient_buffer.get_param(operation.inputs)
            
            #  Execute the backward for the operation.
            gradients = operation.backward_operation(grad_of_output, *operation.inputs)

            #  Update the gradients for the inputs based on tracking type.
            for i, (input_data, gradient) in enumerate(zip(operation.inputs, gradients)):
                if operation.gradients_to_update[i] is not None:
                    # Update externally tracked gradients.
                    operation.gradients_to_update[i] += gradient
                else:
                    # Update internally tracked gradients in the GradientBuffer.
                    self.gradient_buffer.update_param(input_data, gradient)
        self.operation_list = []

    def zero_grad(self):
        self.gradient_buffer.clear()
        self.operation_list = []  



