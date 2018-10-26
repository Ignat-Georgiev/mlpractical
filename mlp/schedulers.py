# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np
import math


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate


class CosineAnnealingWithWarmRestarts(object):
    """Cosine annealing scheduler, implemented as in https://arxiv.org/pdf/1608.03983.pdf"""

    def __init__(self, min_learning_rate, max_learning_rate, total_iters_per_period, max_learning_rate_discount_factor,
                 period_iteration_expansion_factor):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_learning_rate: The minimum learning rate the scheduler can assign
        :param max_learning_rate: The maximum learning rate the scheduler can assign
        :param total_epochs_per_period: The number of epochs in a period
        :param max_learning_rate_discount_factor: The rate of discount for the maximum learning
            rate after each restart i.e. how many times smaller the max learning rate will be
            after a restart compared to the previous one
        :param period_iteration_expansion_factor: The rate of expansion of the period epochs.
            e.g. if it's set to 1 then all periods have the same number of epochs, if it's larger
            than 1 then each subsequent period will have more epochs and vice versa.
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.total_epochs_per_period = total_iters_per_period

        self.max_learning_rate_discount_factor = max_learning_rate_discount_factor
        self.period_iteration_expansion_factor = period_iteration_expansion_factor
        self.curr_epoch = 0
        self.actual_epoch = 0

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """

        # print("epoch_number is", epoch_number, "actual epoch is", self.actual_epoch)
        # check if we are not in a continuous epoch
        if epoch_number != self.actual_epoch:
            self.cont_epoch(epoch_number)
        self.actual_epoch += 1

        # update period if we have reached the end of it
        if self.curr_epoch == self.total_epochs_per_period:
            self.curr_epoch = 0
            self.total_epochs_per_period *= self.period_iteration_expansion_factor
            self.max_learning_rate *= self.max_learning_rate_discount_factor

        # update learning rate based on ugly formula (7)
        # https://arxiv.org/pdf/1711.05101.pdf
        learning_rule.learning_rate = self.min_learning_rate + \
            0.5*(self.max_learning_rate - self.min_learning_rate) * \
            (1 + np.cos(math.pi*self.curr_epoch/self.total_epochs_per_period))

        self.curr_epoch += 1

        return learning_rule.learning_rate

    def cont_epoch(self, epoch_number):
        """Called if some epochs were skipped. Updates all the annealing
        variables based on the number of skipped epochs.

        Args:
            epoch_number: Integer index of training epoch about to be run.
        Updates:
            actual_epoch
            curr_epoch
            max_learning_rate
            total_epochs_per_period
        """

        skipped = int(epoch_number - self.actual_epoch)

        # recursively go through the skipped epochs and update parameters
        while epoch_number != self.actual_epoch:
            diff = self.total_epochs_per_period - self.curr_epoch
            if skipped >= diff:
                skipped -= diff
                self.actual_epoch += diff
                self.total_epochs_per_period *= self.period_iteration_expansion_factor
                self.max_learning_rate *= self.max_learning_rate_discount_factor
                self.curr_epoch = 0
            else:
                skipped -= diff
                self.actual_epoch += diff
                self.curr_epoch += diff
            # print("Stuck in loop, epoch number is", epoch_number, "actual epoch is", self.actual_epoch)

        self.actual_epoch = int(self.actual_epoch)

        assert self.actual_epoch == epoch_number,\
            "Failed to reach the correct epoch number"


class CosineAnnealingWithWarmRestartsPlus(object):
    """Cosine annealing scheduler, implemented as in https://arxiv.org/pdf/1608.03983.pdf"""

    def __init__(self, min_learning_rate, max_learning_rate, total_iters_per_period, max_learning_rate_discount_factor,
                 period_iteration_expansion_factor):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_learning_rate: The minimum learning rate the scheduler can assign
        :param max_learning_rate: The maximum learning rate the scheduler can assign
        :param total_epochs_per_period: The number of epochs in a period
        :param max_learning_rate_discount_factor: The rate of discount for the maximum learning
            rate after each restart i.e. how many times smaller the max learning rate will be
            after a restart compared to the previous one
        :param period_iteration_expansion_factor: The rate of expansion of the period epochs.
            e.g. if it's set to 1 then all periods have the same number of epochs, if it's larger
            than 1 then each subsequent period will have more epochs and vice versa.
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.total_epochs_per_period = total_iters_per_period

        self.max_learning_rate_discount_factor = max_learning_rate_discount_factor
        self.period_iteration_expansion_factor = period_iteration_expansion_factor
        self.curr_epoch = 0
        self.actual_epoch = 0

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """

        # print("epoch_number is", epoch_number, "actual epoch is", self.actual_epoch)
        # check if we are not in a continuous epoch
        if epoch_number != self.actual_epoch:
            self.cont_epoch(epoch_number)
        self.actual_epoch += 1

        # update period if we have reached the end of it
        if self.curr_epoch == self.total_epochs_per_period:
            self.curr_epoch = 0
            self.total_epochs_per_period *= self.period_iteration_expansion_factor
            self.max_learning_rate *= self.max_learning_rate_discount_factor
            learning_rule.reset()  # experimental

        # update learning rate based on ugly formula (7)
        # https://arxiv.org/pdf/1711.05101.pdf
        learning_rule.learning_rate = self.min_learning_rate + \
            0.5*(self.max_learning_rate - self.min_learning_rate) * \
            (1 + np.cos(math.pi*self.curr_epoch/self.total_epochs_per_period))

        self.curr_epoch += 1

        return learning_rule.learning_rate

    def cont_epoch(self, epoch_number):
        """Called if some epochs were skipped. Updates all the annealing
        variables based on the number of skipped epochs.

        Args:
            epoch_number: Integer index of training epoch about to be run.
        Updates:
            actual_epoch
            curr_epoch
            max_learning_rate
            total_epochs_per_period
        """

        skipped = int(epoch_number - self.actual_epoch)

        # # recursively go through the skipped epochs and update parameters
        # while epoch_number != self.actual_epoch:
        #     diff = self.total_epochs_per_period - self.curr_epoch
        #     if skipped >= diff:
        #         skipped -= diff
        #         self.actual_epoch += diff
        #         self.total_epochs_per_period *= self.period_iteration_expansion_factor
        #         self.max_learning_rate *= self.max_learning_rate_discount_factor
        #         self.curr_epoch = 0
        #     else:
        #         skipped -= diff
        #         self.actual_epoch += diff
        #         self.curr_epoch += diff
        #     # print("Stuck in loop, epoch number is", epoch_number, "actual epoch is", self.actual_epoch)
        #
        # self.actual_epoch = int(self.actual_epoch)
        #
        # assert self.actual_epoch == epoch_number,\
        #     "Failed to reach the correct epoch number"



