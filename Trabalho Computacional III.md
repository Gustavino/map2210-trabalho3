# Trabalho Computacional III - MAP2210 - Aplicações de Álgebra Linear  

## Determinação de autovalores e autovetores em matrizes SPD

> O critério de convergência utilizado em todo o desenvolvimento foi, em duas diferentes implementações, o da norma infinita.  

*1*. Norma infinita do vetor resultante da diferença entre dois vetores _x_ e _XO_ através da biblioteca numpy:  

```python
np.linalg.norm(x - XO, np.inf)
```

*2*. Função _infinite_norm_, aplicada sobre um vetor que retorna o valor de sua norma infinita e a posição do valor em sua estrutura:  
```python
import numpy as np


def infinite_norm(vector):
    
    max_element_index = np.argmax(vector)
    min_element_index = np.argmin(vector)
    
    if abs(vector[min_element_index]) == abs(vector[max_element_index]):
        if min_element_index < max_element_index:
            return min_element_index, vector[min_element_index]
        return max_element_index, vector[max_element_index]
    
    if abs(vector[min_element_index]) > abs(vector[max_element_index]):
        return min_element_index, vector[min_element_index]
    return max_element_index, vector[max_element_index]
```

>a. Escolha de 10 matrizes SPD.  

1. **bcsstk03**  
Dimensão: 112x112  
Esparsidade: 94%  
!["bcsstk03"](matrices/images/bcsstk03.png)  
2. **bcsstk05**  
Dimensão: 153x153  
Esparsidade: 89.6%  
!["bcsstk05"](matrices/images/bcsstk05.png)  
3. **mesh2em5**  
Dimensão: 306x306  
Esparsidade: 97.8%  
!["mesh2em5"](matrices/images/mesh2em5.png)  
4. **bcsstk34**  
Dimensão: 588x588  
Esparsidade: 93.8%  
!["bcsstk34"](matrices/images/bcsstk34.png)  
5. **msc01050**  
Dimensão: 1050x1050  
Esparsidade: 97%  
!["msc01050"](matrices/images/msc01050.png)  
6. **plat1919**  
Dimensão: 1919x1919  
Esparsidade: 99%  
!["plat1919"](matrices/images/plat1919.png)  
7. **nasa2910**  
Dimensão: 2910x2910  
Esparsidade: 97.94%  
!["nasa2910"](matrices/images/nasa2910.png)  
8. **sts4098**  
Dimensão: 4098x4098  
Esparsidade: 99.56%  
!["sts4098"](matrices/images/sts4098.png)  
9. **s3rmt3m3**  
Dimensão: 5357x5357  
Esparsidade: 99.27%  
!["s3rmt3m3"](matrices/images/s3rmt3m3.png)  
10. **Muu**  
Dimensão: 7102x7102  
Esparsidade: 99.66%  
!["Muu"](matrices/images/Muu.png)  

>b. Implementação do método da potência e obtenção dos raios espectrais.

```python
import numpy as np
import infinite_norm
import vector_normalization 

def power_method_coo(input_matrix, arbitrary_non_null_vector, tolerance, max_iterations):
    eigenvector = arbitrary_non_null_vector
    greatest_element_index, greatest_element = infinite_norm(eigenvector)
    eigenvector = vector_normalization(eigenvector, greatest_element)

    for current_iteration in range(max_iterations):
        next_iteration_eigenvector = input_matrix.dot(eigenvector)
        eigenvalue = next_iteration_eigenvector[greatest_element_index]
        greatest_element_index, greatest_element = infinite_norm(next_iteration_eigenvector)

        if abs(greatest_element) < 10 ** -18:
            print("The input matrix has the eigenvalue 0, select a new arbitrary vector and restart.")
            return 0, np.empty(len(input_matrix.todense()))

        error = np.linalg.norm(eigenvector - np.dot(next_iteration_eigenvector, 1 / greatest_element), np.inf)
        eigenvector = np.dot(next_iteration_eigenvector, 1 / greatest_element)  # Eigenvector

        if error < tolerance:
            return eigenvalue, eigenvector

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")
```

* Acima, o método da potência emprega como auxiliares as seguintes funções:  
1. _infinite_norm_, descrita no início do relatório.
2. _vector_normalization_, que, nesse caso, normaliza o vetor em relação ao seu maior elemento, isto é, todos os seus componentes são divididos pelo módulo do maior componente.
```python
def vector_normalization(vector, element):
    return np.dot(1 / element, vector)
```  

#### Tempo demandado pelos cálculos dos raios espectrais das matrizes  

##### Log:  
```python
-------- MATRIZ: bcsstk03 --------
Dimensão: 112
Tempo necessário para encontrar o raio espectral: 0.01s


-------- MATRIZ: bcsstk05 --------
Dimensão: 153
Tempo necessário para encontrar o raio espectral: 0.02s


-------- MATRIZ: mesh2em5 --------
Dimensão: 306
Tempo necessário para encontrar o raio espectral: 0.02s


-------- MATRIZ: bcsstk34 --------
Dimensão: 588
Tempo necessário para encontrar o raio espectral: 0.05s


-------- MATRIZ: msc01050 --------
Dimensão: 1050
Tempo necessário para encontrar o raio espectral: 0.04s


-------- MATRIZ: plat1919 --------
Dimensão: 1919
Tempo necessário para encontrar o raio espectral: 0.04s


-------- MATRIZ: nasa2910 --------
Dimensão: 2910
Tempo necessário para encontrar o raio espectral: 0.04s


-------- MATRIZ: sts4098 --------
Dimensão: 4098
Tempo necessário para encontrar o raio espectral: 0.03s


-------- MATRIZ: s3rmt3m3 --------
Dimensão: 5357
Tempo necessário para encontrar o raio espectral: 5.98s


-------- MATRIZ: Muu --------
Dimensão: 7102
Tempo necessário para encontrar o raio espectral: 8.67s
```

##### Gráficos - Tempo x Dimensão da matriz: 
* Oito primeiras matrizes, em ordem crescente pela dimensão:  
    !["Eight matrices - Power Method"](plots/power-method-dimensions-8.png)  
    
* Todas as matrizes:  
    !["All matrices - Power Method"](plots/power-method-dimensions-10.png)  

* Acerca da variação sem padrão aparente nas oito primeiras matrizes, será utilizada a seguinte animação na explicação:  
!["Power Iteration animation"](matrices/gifs/Animation_of_the_Power_Iteration_Algorithm.gif)  
  
    A aproximação inicial do autovetor utilizado na execução do método da potência para todas as matrizes foi simplesmente
  um vetor com 1 em todas as entradas, o que satisfaz as necessidades do vetor não ser nulo e ter norma infinita igual a um.  
    Dado esse detalhe de implementação, os resultados podem variar muito, como a animação acima explicita. Isto é,
    o vetor (1, 1, ... , 1) com certeza é mais próximo do autovetor da matriz _sts4098_ do que do autovetor da 
    matriz _bcsstk34_ e esse fato faz com que a convergência para uma matriz de *dimensão 4098* leve metade do tempo da
    convergência para uma de *dimensão 588*.  
    
    > Matrizes: sts4098 vs bcsstk34  
    Dimensões: 4098 vs 588  
    Tempo para convergência: 0.267s vs 0.512s  

> c. Utilização do método da potência inverso como otimizador do método da potência.

```python
import numpy as np
import infinite_norm
import vector_normalization 

from numpy.linalg import LinAlgError

def inverse_power_method(input_matrix, eigenvector_approximation, tolerance, max_iterations):
    input_matrix_dimension = len(input_matrix)

    transpose_product = np.dot(eigenvector_approximation, input_matrix)
    rayleigh_quotient = np.dot(transpose_product, eigenvector_approximation) / np.dot(eigenvector_approximation,
                                                                                      eigenvector_approximation)

    greatest_element_index, greatest_element = infinite_norm(eigenvector_approximation)
    eigenvector_approximation = vector_normalization(eigenvector_approximation, greatest_element)

    for current_iteration in range(max_iterations):
        rayleigh_identity = np.dot(rayleigh_quotient, np.identity(input_matrix_dimension))
        auxiliary_matrix = np.subtract(input_matrix, rayleigh_identity)

        try:
            next_iteration_eigenvector = np.linalg.solve(auxiliary_matrix, eigenvector_approximation)
        except LinAlgError:
            print("y does not have unique solution, so {} is an eigenvalue".format(rayleigh_quotient))
            return rayleigh_quotient, np.empty(0)

        eigenvalue = next_iteration_eigenvector[greatest_element_index]
        greatest_element_index, greatest_element = infinite_norm(next_iteration_eigenvector)

        next_iteration_eigenvector_normalized = np.dot(next_iteration_eigenvector, 1 / greatest_element)
        error_index, error = infinite_norm(np.subtract(eigenvector_approximation, next_iteration_eigenvector_normalized))
        eigenvector_approximation = np.dot(next_iteration_eigenvector, 1 / greatest_element)

        if error < tolerance:
            eigenvalue = 1 / eigenvalue + rayleigh_quotient
            return eigenvalue, eigenvector_approximation

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")
```

* O método da potência será alterado para delegar ao método da potência inverso o processo de encontrar o raio espectral.  
A partir do momento em que o erro calculado valer 10^-4, o método da potência inversa calcula em poucas iterações o autovalor e o autovetor correspondente utilizando como aproximação inicial o autovetor gerado pelo método da potência.  
É importante notar que, embora o método da potência convirja em um número menor de iterações, ele realiza cálculos computacionalmente caros, como, por exemplo, a inversa de uma matriz.  
Seria como se o método da potência inversa assumisse o processo no meio da [animação](#grficos---tempo-x-dimenso-da-matriz) mostrada anteriormente.  

* Método da potência otimizado:  
```python
import numpy as np
import infinite_norm
import vector_normalization 


def power_method_with_inverse_iteration(input_matrix, arbitrary_non_null_vector, tolerance, max_iterations):
    eigenvector = arbitrary_non_null_vector
    greatest_element_index, greatest_element = infinite_norm(eigenvector)
    eigenvector = vector_normalization(eigenvector, greatest_element)

    for current_iteration in range(max_iterations):
        matrix_eigenvector_product = input_matrix.dot(eigenvector)
        greatest_element_index, greatest_element = infinite_norm(matrix_eigenvector_product)

        if abs(greatest_element) < 10 ** -18:
            print("The input matrix has the eigenvalue 0, select a new arbitrary vector and restart.")
            return 0, np.empty(len(input_matrix.todense()))

        error = np.linalg.norm(eigenvector - np.dot(matrix_eigenvector_product, 1 / greatest_element), np.inf)
        eigenvector = np.dot(matrix_eigenvector_product, 1 / greatest_element)

        if error < 10 ** -4:
            eigenvalue, eigenvector = inverse_power_method(np.copy(input_matrix.todense()), eigenvector,
                                                           tolerance, max_iterations)
            return eigenvalue, eigenvector

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")
```  

#### Tempo necessário aos cálculos dos raios espectrais das matrizes utilizando o método da potência otimizado

##### Log:  

```python
-------- MATRIZ: bcsstk03 --------
Dimensão: 112
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 0.00s


-------- MATRIZ: bcsstk05 --------
Dimensão: 153
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 0.01s


-------- MATRIZ: mesh2em5 --------
Dimensão: 306
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 0.01s


-------- MATRIZ: bcsstk34 --------
Dimensão: 588
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 0.02s


-------- MATRIZ: msc01050 --------
Dimensão: 1050
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 0.14s


-------- MATRIZ: plat1919 --------
Dimensão: 1919
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 0.28s


-------- MATRIZ: nasa2910 --------
Dimensão: 2910
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 0.78s


-------- MATRIZ: sts4098 --------
Dimensão: 4098
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 1.48s


-------- MATRIZ: s3rmt3m3 --------
Dimensão: 5357
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 5.97s


-------- MATRIZ: Muu --------
Dimensão: 7102
Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como otimizador: 7.56s
``` 

* Como fica claro pelos logs, o método da potência inversa como otimizador nem sempre garante melhoria no desempenho. 
Devido ao fato de sua execução demandar muitas operações de ponto flutuante (flops), o custo-benefício de sua utilização 
para matrizes pequenas é baixo, como fica claro no desempenho quase igual ao método da potência puro.  
Entretanto, para matrizes de maior dimensão, como no caso da Muu (7102x7102), os complexos cálculos do método inverso vencem sobre
um número muito maior de iterações do método puro.  
>Matriz: Muu  
>Dimensão: 7102x7102  
>Tempo no método da potência: 8.67s  
>Tempo no método da potência inversa: 7.56s  
>Ganho de desempenho: 13%  

##### Gráficos - Tempo x Dimensão da matriz:  
!["All matrices - Optimized Power Method"](plots/power-method-with-inverse-iteration-graph-by-dimension.png)  
     
>d. Código e aplicação do algoritmo QR para encontrar todos os autovalores de uma matriz quadrada  

```python
import numpy as np


def qr_iteration_eigenvalues(A, precision):
    previous_iteration = A[0][0] + 1

    while abs(previous_iteration - A[0][0]) > precision:
        previous_iteration = A[0][0]
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

    return np.diag(A)
```

#### Tempo para calcular todos os autovalores das matrizes  

##### Log:  

```python
-------- MATRIZ: bcsstk03 --------
Quantidade de autovalores: 112
Tempo necessário para encontrar todos os autovalores: 0.21s


-------- MATRIZ: bcsstk05 --------
Quantidade de autovalores: 153
Tempo necessário para encontrar todos os autovalores: 1.03s


-------- MATRIZ: mesh2em5 --------
Quantidade de autovalores: 306
Tempo necessário para encontrar todos os autovalores: 2.06s


-------- MATRIZ: bcsstk34 --------
Quantidade de autovalores: 588
Tempo necessário para encontrar todos os autovalores: 14.96s


-------- MATRIZ: msc01050 --------
Quantidade de autovalores: 1050
Tempo necessário para encontrar todos os autovalores: 57.39s


-------- MATRIZ: plat1919 --------
Quantidade de autovalores: 1919
Tempo necessário para encontrar todos os autovalores: 382.30s


-------- MATRIZ: nasa2910 --------
Quantidade de autovalores: 2910
Tempo necessário para encontrar todos os autovalores: 125.18s


-------- MATRIZ: sts4098 --------
Quantidade de autovalores: 4098
Tempo necessário para encontrar todos os autovalores: 294.78s


-------- MATRIZ: s3rmt3m3 --------
Dimensão: 5357
Não foi possível determinar os autovalores, o tempo para convergência superou nove horas de processamento.


-------- MATRIZ: Muu --------
Dimensão: 7102
Não foi possível determinar os autovalores, o tempo para convergência superou nove horas de processamento.
```

##### Gráfico - Tempo x Dimensão da matriz:  
!["QR Algorithm - Eight smallest matrices"](plots/qr-algorithm.png)  

* A convergência mais rápida de matrizes maiores em relação a matrizes menores se justifica na dependência da
 velocidade de convergência com a diferença de tamanho entre autovalores consecutivos da matriz. Ou seja, a matriz 
 _nasa2910_ converge mais rapidamente que a matriz _plat1919_ (mesmo tendo uma dimensão 1.5x maior) porque seus autovalores 
 consecutivos são menores e resulta num processo mais rápido para encontrar seu espectro completo.

* Como explicado [nessas notas de aula](http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf), o algoritmo QR não é
recomendado para matrizes muito grandes, devido à necessidade de trabalhar com matrizes densas e o excesso de memória e 
processamento que seus cálculos demandariam. Assim, o algoritmo provou-se imensamente demorado para grandes matrizes esparsas,
visto que ultrapassou nove horas de processamento para as duas matrizes com dimensão maior que 5000 deste estudo.

* Seria possível acelerar o algoritmo QR utilizando o método de Householder para transformar uma matriz cheia em uma
matriz do formato Hessenberg, entretanto, isso foge ao escopo deste trabalho.

* O algoritmo QR implementado utiliza como fator de parada uma diferença menor que um dado ε (precision). 
A diferença é determinada pelo primeiro elemento de cada diagonal, este o maior autovalor, entre a última iteração e a iteração anterior.  
Dessa forma, o invariante (um tanto informal) do loop _while_ contido no algoritmo é definido como:
> "À medida que uma matriz quadrada A se aproxima de uma matriz triangular superior, triangular inferior ou diagonal, 
os elementos da diagonal da matriz A sâo aproximações para os autovalores de A".  
    1. Inicialização: toda matriz SPD A contém em sua diagonal principal elementos (a1, a2, ..., an) que, 
    através de prova por contradição (e indução, no caso numérico), se aproximam dos autovalores de A conforme o 
    determinante de A passa a depender somente da diagonal principal. Isto é, ao passo que A passar a apresentar as 
    características contidas no invariante.  
    2. Manutenção:  a cada iteração, os elementos abaixo da diagonal principal de A se aproximam de zero e os elementos em sua diagonal
    se aproximam dos autovalores.  
    3. Encerramento:  A é uma matriz triangular superior e contém seus autovalores em sua diagonal principal.

>e. Discussão dos resultados  
*  Finalmente, ao longo do relatório foram demonstradas estratégias para encontrar autovalores e autovetores, junto de 
suas vantagens e limitações de implementações. Portanto, é possível fazer as seguintes observações:  
    * O algoritmo QR puro é claramente não recomendado, aplicações utilizando a transformação de Householder seguidas de
    tridiagonalização da matriz são essenciais para diminuir o tempo de execução e, dessa forma, dar viabilidade prática
    ao algoritmo.  
    * O algoritmo da potência otimizado com o algoritmo da potência inversa pode ser muito útil em grandes matrizes, podendo 
    ainda ser otimizado com, por exemplo, o método de Aitken.