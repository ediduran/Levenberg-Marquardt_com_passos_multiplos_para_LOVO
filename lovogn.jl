using LinearAlgebra
using DelimitedFiles
using Printf

include("./modelos.jl")
include("./auxiliar.jl")

"""
Objetivo: Determinar o melhor ajuste para o modelo, via Método de
Levenberg-Marquardt Modificado, aplicado ao problema LOVO.

Entrada:
modelo -  modelo a ser ajustado.
deparcial - derivadas parciais do modelo.
x - vetor com os parâmetros iniciais.
confiabilidade - confiabilidade.
λ - parametro LM.
ε - tolerância máxima.
kmax - número máximo de iterações.
γ - parametro para busca linear.
problema - nome do problema
verbose - Imprime informações acerca dos iterandos.
Saída:
x - parâmetros do modelo após o ajuste.
norma_grad - norma do gradiente.
k - número de iterações.
"""
function lovogn!(modelo, deparcial, x, confiabilidade, λ, ε, kmax, γ, problema::String, verbose = false)

    #Extrair dados
    D = readdlm("problemas/$(problema)/dados.dat")
    t = copy(D[:,1])
    y = copy(D[:,2])

    itime = time()
    
    r = length(t)
    p = Int((r * confiabilidade) / 100)
    n = length(x)
    k = 0
    l = 0
    dk = zeros(n)
    Aux = zeros(n,n)

    choj = CholeskyPivoted{Float64,Array{Float64,2}}
    
    #Determinar prop. de Ci
    dados = resjac(modelo, deparcial, x, t, y, r, p)
    l = l + 1
    mgrad = - dados[1]' * dados[2]
    norma_grad = norm(mgrad)

    while norma_grad > ε

        Aux = dados[1]' * dados[1]
        try
            choj = cholesky(Aux, Val(true))
        catch
            for i=1:n
                Aux[i, i] = Aux[i,i] + λ
            end
            choj = cholesky(Aux, Val(true))
        end
       
        ldiv!(dk, choj, mgrad)

        h = 1.0
        while sum((modci(modelo, x + h * dk, t, y, p, dados[4])).^(2.0)) > (dados[3] - γ * h * mgrad' * dk )
            l = l + 1
            h = h / 2.0
        end

        x = x + h * dk

        k = k + 1
        if k > kmax
            @printf("%s\n", "----------------------- ATENÇÃO! ------------------------")
            @printf("%s\n", "--------- Excedeu o número máximo de iterações! ---------")
            break
        end

        dados = resjac(modelo, deparcial, x, t, y, r, p)
        l = l + 1
        mgrad = - dados[1]' * dados[2]
        norma_grad = norm(mgrad)
        
        if verbose == true
            @printf("%s\n", "-----------------------------------------------------------")
            @printf("Iteração: %s\n", k)
            @printf("Número de avaliações da função: %s\n", l)
            @printf("Parametros calculados: %s\n", x)
            @printf("Valor de Sp(x): %s\n", dados[3])
            @printf("Norma do gradiente: %s\n", norma_grad)
        end
        
    end

    ftime = time()

  
    @printf("%s\n", "-----------------------------------------------------------")
    @printf("Iteração: %s\n", k)
    @printf("Número de avaliações da função: %s\n", l)
    @printf("Parametros calculados: %s\n", x)
    @printf("Valor de Sp(x): %s\n", dados[3])
    @printf("Norma do gradiente: %s\n", norma_grad)
    @printf("tempo: %s\n", ftime - itime)
    @printf("%s\n", "-----------------------------------------------------------")

    return x, norma_grad, k

end

lovogn!(sen02!, jacsen02!, [5.0, 5.0, 5.0, 5.0, 5.0], 90, 0.1, 10.0^(-4.0), 500, 10.0^(-4.0), "sen02_5000")

GC.gc()