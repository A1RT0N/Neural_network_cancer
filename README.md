# Classificação de Câncer de Lábio - IC FAPESP

## Objetivo
Desenvolver uma rede neural para classificação de câncer de lábio, com foco em avaliar diferentes arquiteturas e técnicas de data augmentation para melhorar os resultados. A tarefa é seguir uma abordagem experimental para definir qual arquitetura e configuração trazem os melhores resultados.

## Estrutura de Diretórios
A estrutura de arquivos está organizada da seguinte maneira:

- **efficient**, **inception**, **mobilenet**, **resnet**, **unet**, **vgg19**  
  Cada diretório contém duas subpastas:
  - **baseline**: Modelos base (sem augmentação de dados).
  - **data_augmentation**: Modelos com augmentação de dados.


## Passos para Implementação

### 1. Treinamento Inicial (Baseline)
- **Objetivo**: Treinar os modelos de classificação com os dados brutos (sem augmentação de dados e sem balanceamento de classes).
- **Arquiteturas a serem testadas**: 
  - EfficientNet
  - Inception
  - MobileNet
  - ResNet
  - U-Net
  - VGG19

### 2. Teste com Data Augmentation
- **Objetivo**: Melhorar a performance com augmentação de dados, inicialmente sem balanceamento de classes.
- **Técnicas de Augmentation**:
  - Flip horizontal e vertical
  - Random crop
  - Aumentos geométricos (rotação, dropout de pixels)
  - Aumentos espectrais (alteração na ordem dos canais, variação de tons de cinza)

### 3. Teste com Balanceamento de Classes
- **Objetivo**: Aplicar técnicas de undersampling na classe majoritária e data augmentation em todas as classes.
- **Considerações**:
  - Foco em aumentos geométricos que não prejudiquem a visualização das lesões.

### 4. Avaliação de Resultados
- **Análise Inicial**: Comparar os resultados dos testes com baseline e com augmentation.
- **Arquitetura Promissora**: Com base nos resultados iniciais, escolher a arquitetura mais promissora para otimização de hiperparâmetros e ajustes finos.

### 5. Ajustes Finais
- **Igualdade nos Hiperparâmetros**: Ao testar diferentes arquiteturas, manter os mesmos hiperparâmetros (como seed do `torch/numpy/random`) para garantir uma comparação justa.

## Conclusão
Com esses testes, será possível comparar o desempenho das diferentes arquiteturas e abordagens de augmentação de dados. Isso permitirá a escolha da melhor configuração para a classificação de câncer de lábio, com base nas necessidades do projeto.
