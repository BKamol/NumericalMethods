import numpy as np
from typing import List


class SimplexMethod:
    """
    Class for solving canonic optimization task
    c.T@x -> min
    A@x = P0 (by default, can be changed by setting inequalities)
    x >= 0 (by default, can be changed by setting negative=True)
    """
    def __init__(self, c: List[float], A: List[List[float]], P0: List[float], inequalities: List[int]=None, negative: bool=False) -> None:
        """
        c - coefficients of linear function,
        A - matrix of boundaries
        P0 - right side of boundaries
        inequalities - rows with inequality (0 means equality in a row, 1 means <= in a row, -1 means >= in a row)
        """
        self.c = np.array(c)
        self.A = np.array(A)
        self.P0 = np.array(P0)
        self.inequalities = np.array(inequalities) if inequalities else None
        self.negative = negative


    def positive_delta(self, delta: List[float]) -> int:
        """
        Takes array of estims delta and returns index of a positive delta
        """
        for i in range(1, len(delta)):
            if delta[i] > 0:
                return i
        return -1
    
    def is_function_unbounded(self, new_basis_index: int):
        """
        Checks if function is unbounded
        (If there is positive delta for which every matrix coefficient in same column is nonpositive)
        """
        return all(self.simplexMatrix[:, new_basis_index] <= 0)
    
    def find_replacable_basis(self, new_basis_index):
        """
        Finds index of vector that will leave basis
        (Finds i: s[i][0] / s[i][new_vector] is minimized, s is simplexMatrix)
        """
        rows = self.simplexMatrix.shape[0]
        replacable_basis_index = -1
        min_relation = 1e15
        for i in range(rows):
            if self.simplexMatrix[i, new_basis_index] <= 0: continue
            # print(min_relation, replacable_basis_index)
            relation = self.simplexMatrix[i, 0] / (self.simplexMatrix[i, new_basis_index] + 1e-15 )
            if 0 < relation and relation < min_relation:
                min_relation = relation
                replacable_basis_index = i
        return replacable_basis_index
    
    def recalculating_simplex_matrix(self, replacable_basis_index, new_basis_index):
        """
        Recalculates simples matrix using Gauss' formula
        """
        self.simplexMatrix[replacable_basis_index, :] = self.simplexMatrix[replacable_basis_index, :] / self.simplexMatrix[replacable_basis_index, new_basis_index]
        for i in range(len(self.simplexMatrix)):
            if i == replacable_basis_index: continue
            self.simplexMatrix[i, :] = self.simplexMatrix[i, :] - self.simplexMatrix[replacable_basis_index, :] * self.simplexMatrix[i, new_basis_index]
    
    def is_artificial_vector_in_basis(self, B):
        additional_vectors_count = np.sum(np.abs(self.inequalities)) if self.inequalities is not None else 0
        notartif_vectors_count = self.A.shape[1] if not self.negative else 2*self.A.shape[1]
        if max(B) > notartif_vectors_count + additional_vectors_count:
            # print(B, self.A.shape[1])
            print("Artificial vector in basis => domain is empty")

    def add_variables_for_inequalities(self, A, c):
        n = self.A.shape[0]
        for i in range(len(self.inequalities)):
            sign = self.inequalities[i]
            if sign == 0: continue
            new_vector = [0 for _ in range(n)]
            new_vector[i] = sign
            A = np.append(A, np.array(new_vector).reshape((n, 1)), 1)
            c = np.append(c, [0])
        return A, c
    
    def build_simplex_matrix(self, A, c):
        rows, cols = A.shape
        n = self.A.shape[0]
        B = [i for i in range(cols - n, cols)]
        cb = c[B]
        self.simplexMatrix = np.append(self.P0.reshape((self.A.shape[0], 1)), A, 1)
        c0 = np.append([0], c)
        alpha = -c0
        beta = cb@self.simplexMatrix
        delta = alpha + beta
        return B, alpha, beta, delta


    def solve(self):
        """
        Solves task by building simplex matrix, basis, deltas and iterating until all deltas are nonpositive
        returns (corrdinates of solution, minimal value, indexes of vectors in basis)
        """
        n = self.A.shape[0]
        A = self.A
        c = self.c

        # splitting negative variables, negative variable with index i will be split to indexes 2*i and 2*i+1
        if self.negative:
            for j in range(0, self.A.shape[1]):
                i = 2*j
                old_vector = A[:, i].reshape((n, 1))
                A = np.append(np.append(A[:, :i+1], -old_vector, 1), A[:, i+1:], 1)
                c = np.append(np.append(c[:i+1], -c[i]), c[i+1:])


        # adding variables for inequalities
        if self.inequalities is not None:
            A, c = self.add_variables_for_inequalities(A, c)

        # adding variables for identity basis
        A = np.append(A, np.eye(n), 1)
        c = np.append(c, np.array([1e15 for _ in range(n)]))

        #building simplex matrix
        B, alpha, beta, delta = self.build_simplex_matrix(A, c)

        #Iterating by replacing basis vectors
        new_basis_index = self.positive_delta(delta)
        while new_basis_index != -1:
            if self.is_function_unbounded(new_basis_index):
                print("Function is unbounded")
                return self.simplexMatrix[:, 0], alpha[0], B
            
            replacable_basis_index = self.find_replacable_basis(new_basis_index)
            B[replacable_basis_index] = new_basis_index

            # print(new_basis_index, replacable_basis_index)
            # print(B)
            # print(delta)
            # print(deltav)
            # Recalculating simplexMatrix and delta with Gauss formula
            self.recalculating_simplex_matrix(replacable_basis_index, new_basis_index)
            alpha = alpha - self.simplexMatrix[replacable_basis_index, :] * alpha[new_basis_index]
            beta = beta - self.simplexMatrix[replacable_basis_index, :] * beta[new_basis_index]
            delta = alpha + beta
            new_basis_index = self.positive_delta(delta)

        #checking answer
        self.is_artificial_vector_in_basis(B)

        answer_coord = self.simplexMatrix[:, 0]
        answer = alpha[0]

        return answer_coord, answer, B
