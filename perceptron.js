// Coluna 1: Tem dinheiro?
// Coluna 2: Pizzaria perto?
// Coluna 3: Está com vontade?

const TabelData = [
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [0, 1, 1, 0],
];

function perceptron(data) {
    let w0 = 0;
    let w1 = 0;
    let w2 = 0;
    let w3 = 0;
    const learningRate = 0.1;
    let check = false;
    let iteration = 0;

    while (!check) {
        let errors = 0;  // Contador de erros

        for (let i = 0; i < data.length; i++) {
            // Calcula a saída do perceptron
            let z = w1 * data[i][0] + w2 * data[i][1] + w3 * data[i][2] + w0;
            z = z > 0 ? 1 : 0;  // Função de ativação

            console.log(`Iteração ${iteration} - Amostra ${i + 1}`);
            console.log(`Para esta amostra temos o Estimado:${z} e correto de ${data[i][3]}`);
            console.log(`-----------------------------------------`);
            console.log(`w0: ${w0}`, `w1: ${w1}`, `w2: ${w2}`, `w3: ${w3}`, `z: ${z}`);

            // Se houver erro, recalcula os pesos
            if (z !== data[i][3]) {
                errors += 1;  // Conta erro

                console.log("Recalculando Pesos");
                w0 += learningRate * (data[i][3] - z);  // Atualiza o bias
                w1 += learningRate * (data[i][3] - z) * data[i][0];
                w2 += learningRate * (data[i][3] - z) * data[i][1];
                w3 += learningRate * (data[i][3] - z) * data[i][2];

                console.log(`Novo w0: ${w0}`, `Novo w1: ${w1}`, `Novo w2: ${w2}`, `Novo w3: ${w3}`);
                console.log(``);
            }
        }

        // Verifica se houve algum erro na iteração
        if (errors === 0) {
            check = true;  // Convergiu
        }

        iteration++;  // Incrementa o contador de iterações para fins de debug
        if (iteration > 1000) break;  // Limite para evitar loops infinitos
    }

    console.log("Perceptron treinado com sucesso!");
}

perceptron(TabelData);
