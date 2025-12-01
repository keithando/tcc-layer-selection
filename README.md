# Projeto de Formatura - Seleção de Camadas para o Ajuste Fino de Grandes Modelos de Linguagem

## Descrição Geral

Este repositório contém a implementação utilizada para a pesquisa em seleção de camadas para o ajuste fino eficiente de grandes modelos de linguagem. Há ambientes e códigos específicos para a seleção e ajuste de modelos *encoder-only* e *decoder-only*, conforme separados na estrutura do repositório.

## Ambiente

As versões específicas de cada ambiente (experimentos *encoder-only* e *decoder-only*) encontram-se nas pastas /envs/decoder e /envs/encoder.

A instalação deve ser feita dentro de um ambiente virtual utilizando o arquivo `requirements.txt`.

Em nossas execuções utilizamos:

**Python:** 3.12.x  
**Sistema Operacional:** Linux 

Para criar o ambiente e ativá-lo:

```bash
python3 -m venv nome_do_ambiente
source /nome_do_ambiente/bin/activate    
```

Com o ambiente ativo, basta executar o seguinte comando para instalar as dependências:
```bash
pip install --upgrade pip
pip install -r envs/xxxxx/requirements.txt
```

## Execução

Com o ambiente configurado, é possível executar os experimentos com os argumentos desejados. Exemplos de execuções encontram-se nos arquivos *shell* na pasta /exemplos/.