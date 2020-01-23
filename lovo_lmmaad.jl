using LinearAlgebra
using DelimitedFiles
using Printf

include("./modelos.jl")
include("./auxiliar.jl")

"""
Objetivo: Determinar o melhor ajuste para o modelo, via Método de
Levenberg-Marquardt Modificado Acelerado Adaptado, aplicado ao problema LOVO.

Entrada:
modelo -  modelo a ser ajustado.
deparcial - derivadas parciais do modelo.
x - vetor com os parâmetros iniciais.
confiabilidade - confiabilidade. 
μ - raio inicial da região de confiança.
q - parâmetros para a ánalise da região de confiança.
αmax - um máximo para α.
ε - tolerância máxima.
kmax - número máximo de iterações.
problema - nome do problema
verbose - Imprime informações acerca dos iterandos.
Saída:
x - parâmetros do modelo após o ajuste.
norma_grad - norma do gradiente.
k - número de iterações.
"""
function lovolmmaad!(modelo, deparcial, x, confiabilidade, μ, M, q, αmax, ε, kmax, problema::String, verbose = true)

    itime = time()

    function lambda(fk, u, d)
        return u * norm(fk) ^ d
    end

    #Extrair dados
    D = readdlm("problemas/$(problema)/dados.dat")
    t = copy(D[:,1])
    y = copy(D[:,2])
    
    r = length(t)
    p = Int((r * confiabilidade) / 100)
    n = length(x)
    k = 0
    l = 0
    λ = 0.0
    α = 0.0
    αk = 0.0
    δk = 0.0
    ρ = 0.0
    aredk = 0.0
    predk = 0.0
    normf = 0.0
    dk = zeros(n)
    dkc = zeros(n)
    sk = zeros(n) 
    yk = zeros(n)
    Aux = zeros(n,n)
    fyk = zeros(p)
    
    flag = 1

    #Determinar prop. de Ci
    dados = resjac(modelo, deparcial, x, t, y, r, p)
    l = l + 1
    mgrad = - dados[1]' * dados[2]
    norma_grad = norm(mgrad)

    while norma_grad > ε

        if flag == 1
            normf = norm(dados[2])
        end

        if normf >= 1
           δk = 1 / normf
        else
            δk = 1 + 1/k 
        end

        if flag == 1
            Aux = dados[1]' * dados[1]
        end
        
        if flag == 0
            for i=1:n
                Aux[i, i] = Aux[i,i] - λ
            end
        end

        λ = lambda(dados[2], μ, δk)

        for i=1:n
            Aux[i, i] = Aux[i,i] + λ
        end
        choj = cholesky(Aux, Val(true))
        ldiv!(dk, choj, mgrad)

        yk .= x .+ dk

        fyk = modci(modelo, yk, t, y, p, dados[4])
        l = l + 1
        jfyk = - dados[1]' * fyk
        ldiv!(dkc, choj, jfyk)

        αk = 1 + (λ * dkc' * dkc) / (dkc' * Aux * dkc)
        if αk < αmax 
            α = αk
        else
            α = αmax
        end

        sk .= dk .+ α * dkc

        fxsk = modci(modelo, x + sk, t, y, p, dados[4])
        aux3 = dados[2]' * dados[2]
        aredk = aux3 - (fxsk' * fxsk)
        aux1 = dados[2] + dados[1] * dk
        aux2 = fyk + dados[1] * dkc
        predk = aux3 - (aux1' * aux1) + (fyk' * fyk) - (aux2' * aux2)

        ρ = aredk / predk

        if ρ > q[1]
            x .= x .+ sk
            flag = 1
        else
            flag= 0
        end

        if ρ < q[2]
            μ = 4.0 * μ
        elseif ρ > q[3]
            μ = max((μ/4.0), M) 
        end

        k = k + 1
        if k > kmax
            @printf("%s\n", "-----------------------------------------------------------")
            @printf("%s\n", "----------------------- ATENÇÃO! ------------------------")
            @printf("%s\n", "--------- Excedeu o número máximo de iterações! ---------")
            break
        end

        if flag == 1
            dados = resjac(modelo, deparcial, x, t, y, r, p)
            l = l + 1
            mgrad = - dados[1]' * dados[2]
            norma_grad = norm(mgrad)
        end

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

lovolmmaad!(sen02!, jacsen02!, [5.0, 5.0, 5.0, 5.0, 5.0], 90, 0.5, 10.0^(-6.0), [0.0001, 0.25, 0.75], 5.0, 10.0^(-4.0), 100, "sen02_1000")

GC.gc()