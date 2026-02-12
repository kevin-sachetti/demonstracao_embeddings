# Sistema de Busca por Similaridade com Embeddings

Sistema de busca semântica utilizando **Sentence Transformers** e **FAISS** para três casos de uso práticos: busca em FAQs, recomendação de filmes e detecção de anomalias em feedbacks.

## Sobre o Projeto

Este projeto demonstra aplicações *básicas e simplificadas* de embeddings e busca por similaridade vetorial, implementando:

1. **Busca Semântica em FAQs**: Encontra respostas relevantes em documentos de perguntas frequentes de diferentes áreas (advocacia, engenharia, meio ambiente, farmácia e exames laboratoriais)

2. **Sistema de Recomendação de Filmes**: Recomenda filmes baseado em descrições em linguagem natural

3. **Detecção de Anomalias**: Identifica feedbacks anômalos analisando similaridade semântica entre avaliações

**OBS: todo o material de "\dados" foi feito artificialmente atraves de IA, recomendo que faça teste com material proprio.**

## Estrutura do Projeto

```
embeddings_project/
├── dados/
│   ├── faq/                    # 5 PDFs com perguntas frequentes
│   ├── filmes/                 # 5 TXTs com descrições de filmes
│   └── avaliacoes/             # JSON com 100 feedbacks
├── embeddings/                 # Embeddings gerados (criado após execução)
│   ├── faq/
│   │   └── faq_embeddings.json
│   ├── filmes/
│   │   └── filmes_embeddings.json
│   └── avaliacoes/
│       └── feedbacks_embeddings.json
├── src/
│   ├── converte_embeddings.py  # Script para gerar embeddings
│   └── busca_similaridade.py   # Interface CLI de busca
├── requirements.txt
└── README.md
```

## Instalação

### Pré-requisitos

- Python 3.9 ou superior
- pip

### Passos

1. Clone ou baixe o repositório

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como Usar

### Passo 1: Gerar os Embeddings

Primeiro, execute o script de conversão para processar todos os documentos e gerar os embeddings:

```bash
python src/converte_embeddings.py
```

Este processo irá:
- Ler os 5 PDFs de FAQ e extrair perguntas e respostas
- Ler as 5 descrições de filmes
- Carregar os 100 feedbacks do JSON
- Gerar embeddings usando o modelo `all-MiniLM-L6-v2`
- Salvar os embeddings no diretório `embeddings/`

**Tempo estimado**: 1-3 minutos (dependendo do hardware)

### Passo 2: Executar o Sistema de Busca

Após gerar os embeddings, execute o sistema de busca interativo:

```bash
python src/busca_similaridade.py
```

Você verá um menu com três opções:

#### 1 Modo FAQ
Digite uma pergunta e o sistema retornará as 3 respostas mais relevantes de todas as FAQs disponíveis, incluindo:
- Score de similaridade
- Fonte (qual FAQ)
- Pergunta e resposta correspondente

**OBS: é importante saber o contexto do material consultado ao perguntar, pois só assim fará sentido as respostas do modelo**

**Exemplo**:
```
 Digite sua pergunta: Como faço para me divorciar?

TOP 3 RESPOSTAS MAIS RELEVANTES
#1 Fonte: faq_advocacia
Pergunta: Quanto tempo demora um processo de divórcio consensual?
Resposta: Em casos de divórcio consensual...
```

#### 2 Modo Filmes
Descreva o tipo de filme que procura e o sistema ranqueará TODOS os 5 filmes por similaridade.

**Exemplo**:
```
Descreva o filme que procura: viagem pelo tempo e espaço

FILMES RANQUEADOS POR SIMILARIDADE
#1 - INTERSTELLAR (2014)
   Em um futuro próximo, a Terra enfrenta...
```

#### 3 Modo Anomalias
Detecta automaticamente os feedbacks mais anômalos (aqueles com menor similaridade média com os demais).

O sistema calcula a similaridade de cada feedback com todos os outros 99 e identifica os 3 mais diferentes.

**Exemplo de saída**:
```
TOP 3 FEEDBACKS ANÔMALOS DETECTADOS

   Anomalia #1
   ID: 98 | Usuário: joana_silva | Data: 2024-03-01
   Score Médio de Similaridade: 0.2341
   Feedback: Pizza de calabresa está com preço...
```

## Tecnologias Utilizadas

- **Sentence Transformers**: Geração de embeddings semânticos
- **FAISS** (Facebook AI Similarity Search): Busca eficiente por similaridade
- **PyPDF**: Extração de texto de arquivos PDF
- **NumPy**: Manipulação de arrays e cálculos matemáticos

## Dados Incluídos

### FAQs (5 PDFs):
- Advocacia (7 perguntas)
- Engenharia Civil (7 perguntas)
- Meio Ambiente (7 perguntas)
- Farmácia Magistral (7 perguntas)
- Laboratório de Exames (7 perguntas)

### Filmes (5 TXTs):
- Interstellar (2014)
- Gravidade (2013)
- A Chegada (2016)
- Matrix (1999)
- Submerso (2020)

### Avaliações (1 JSON):
- 100 feedbacks totais
- 97 feedbacks normais (sobre chatbots/IA)
- 3 feedbacks anômalos (conteúdo não relacionado)

## Configurações

### Modelo de Embeddings

Por padrão, o projeto usa o modelo `all-MiniLM-L6-v2`, que oferece um bom equilíbrio entre qualidade e velocidade. Para alterar o modelo, edite os arquivos Python:

```python
# Em converte_embeddings.py e busca_similaridade.py
conversor = ConversorEmbeddings(modelo='nome-do-modelo')
```

### Threshold de Similaridade

O threshold padrão é 0.6. Para ajustar:

```python
# Em busca_similaridade.py
self.threshold = 0.6  # Valores entre 0 e 1
```

## Metodologia

### Busca Semântica

1. Documentos são convertidos em embeddings (vetores de 384 dimensões)
2. Query do usuário é convertida em embedding
3. Embeddings são normalizados (norma L2 = 1)
4. FAISS calcula similaridade de cosseno via produto interno
5. Resultados são ranqueados por similaridade

### Detecção de Anomalias

1. Para cada feedback, calcula-se a similaridade com todos os outros 99
2. Score médio de similaridade é calculado para cada feedback
3. Feedbacks com menor score médio são considerados anômalos
4. Método baseado em similaridade média entre embeddings

## Casos de Uso

- **Suporte ao Cliente**: Automatizar respostas a perguntas frequentes
- **Sistemas de Recomendação**: Sugerir conteúdo baseado em descrições
- **Moderação de Conteúdo**: Detectar mensagens spam ou off-topic
- **Análise de Feedback**: Identificar avaliações atípicas

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos casos de uso
- Melhorar a documentação

## Licença

MIT License

Copyright (c) 2026 Kevin Sachetti Correa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Autor

**Kevin Sachetti Correa**

- LinkedIn: [linkedin.com/in/kevin-sachetti-correa](https://www.linkedin.com/in/kevin-sachetti-correa/)
- GitHub: [github.com/kevin-sachetti](https://github.com/kevin-sachetti)

Projeto desenvolvido como demonstração prática e simplificada de embeddings e busca por similaridade vetorial.

---

**Dica**: Para melhores resultados, seja específico nas queries e use linguagem natural! De Preferencia com um material que tu conheça previamente e saiba o que perguntar.
