#!/usr/bin/env python3
"""
converte_embeddings.py

Converte documentos (PDFs, TXTs, JSON) em embeddings usando Sentence Transformers
e salva os resultados em arquivos JSON para busca posterior.

Estrutura de saída:
- embeddings/faq/faq_embeddings.json
- embeddings/filmes/filmes_embeddings.json
- embeddings/avaliacoes/feedbacks_embeddings.json
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


class ConversorEmbeddings:
    """Classe responsável por converter documentos em embeddings."""
    
    MODELO_PADRAO = 'all-MiniLM-L6-v2'
    
    def __init__(self, modelo: str = MODELO_PADRAO):
        """
        Inicializa o conversor com o modelo especificado.
        
        Args:
            modelo: Nome do modelo Sentence Transformers a ser usado
        """
        print(f"🔄 Carregando modelo: {modelo}")
        self.modelo = SentenceTransformer(modelo)
        print("✅ Modelo carregado com sucesso!\n")
    
    def extrair_texto_pdf(self, caminho: str) -> str:
        """
        Extrai todo o texto de um arquivo PDF.
        
        Args:
            caminho: Caminho para o arquivo PDF
            
        Returns:
            Texto completo extraído do PDF
        """
        try:
            leitor = PdfReader(caminho)
            texto_completo = '\n'.join(
                pagina.extract_text() for pagina in leitor.pages
            )
            return texto_completo.strip()
        except Exception as erro:
            print(f"❌ Erro ao ler PDF {caminho}: {erro}")
            return ''
    
    def extrair_texto_txt(self, caminho: str) -> str:
        """
        Lê o conteúdo de um arquivo de texto.
        
        Args:
            caminho: Caminho para o arquivo TXT
            
        Returns:
            Conteúdo do arquivo
        """
        try:
            with open(caminho, 'r', encoding='utf-8') as arquivo:
                return arquivo.read().strip()
        except Exception as erro:
            print(f"❌ Erro ao ler TXT {caminho}: {erro}")
            return ''
    
    def carregar_feedbacks_json(self, caminho: str) -> List[Dict]:
        """
        Carrega feedbacks de um arquivo JSON.
        
        Args:
            caminho: Caminho para o arquivo JSON
            
        Returns:
            Lista de dicionários com os feedbacks
        """
        try:
            with open(caminho, 'r', encoding='utf-8') as arquivo:
                dados = json.load(arquivo)
                return dados.get('feedbacks', [])
        except Exception as erro:
            print(f"❌ Erro ao ler JSON {caminho}: {erro}")
            return []
    
    def gerar_embeddings(self, textos: List[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            textos: Lista de strings para converter
            
        Returns:
            Array numpy com os embeddings
        """
        print(f"🔄 Gerando embeddings para {len(textos)} textos...")
        embeddings = self.modelo.encode(textos, show_progress_bar=True)
        print("✅ Embeddings gerados!\n")
        return embeddings
    
    def salvar_embeddings(self, dados: Dict, caminho_saida: str) -> None:
        """
        Salva embeddings e metadados em arquivo JSON.
        
        Args:
            dados: Dicionário com embeddings e metadados
            caminho_saida: Caminho para salvar o arquivo JSON
        """
        caminho = Path(caminho_saida)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        
        with open(caminho, 'w', encoding='utf-8') as arquivo:
            json.dump(dados, arquivo, ensure_ascii=False, indent=2)
        
        print(f"✅ Embeddings salvos em: {caminho_saida}\n")
    
    def extrair_perguntas_respostas_de_texto(self, texto: str) -> tuple:
        """
        Extrai perguntas e respostas de um texto formatado como FAQ.
        
        Detecta perguntas pelo padrão "N." no início (ex: "1.", "2.", "3.").
        Ignora a primeira linha (título do FAQ).
        
        Args:
            texto: Texto completo do FAQ
            
        Returns:
            Tupla com (lista_perguntas, lista_respostas)
        """
        import re
        
        linhas = texto.split('\n')
        
        # Pular título e limpar linhas vazias
        linhas_limpas = []
        pulou_titulo = False
        for linha in linhas:
            linha_limpa = linha.strip()
            if not linha_limpa:
                continue
            if not pulou_titulo:
                pulou_titulo = True
                continue
            linhas_limpas.append(linha_limpa)
        
        perguntas = []
        respostas = []
        
        i = 0
        padrao_numero = re.compile(r'^\d+\.')
        
        while i < len(linhas_limpas):
            linha = linhas_limpas[i]
            
            # Achou início de pergunta?
            if padrao_numero.match(linha):
                # Acumular pergunta até achar linha que termina com ?
                pergunta_partes = [linha]
                i += 1
                
                while i < len(linhas_limpas) and not linhas_limpas[i-1].endswith('?'):
                    pergunta_partes.append(linhas_limpas[i])
                    i += 1
                
                pergunta_completa = ' '.join(pergunta_partes)
                
                # Acumular resposta até próxima pergunta (próximo número) ou fim
                resposta_partes = []
                while i < len(linhas_limpas) and not padrao_numero.match(linhas_limpas[i]):
                    resposta_partes.append(linhas_limpas[i])
                    i += 1
                
                resposta_completa = ' '.join(resposta_partes)
                
                # Só adiciona se ambos não estiverem vazios
                if pergunta_completa.strip() and resposta_completa.strip():
                    perguntas.append(pergunta_completa.strip())
                    respostas.append(resposta_completa.strip())
            else:
                i += 1
        
        return perguntas, respostas
    
    def processar_faqs(self, diretorio_entrada: str, caminho_saida: str) -> None:
        """
        Processa todos os PDFs de FAQ e gera embeddings.
        
        Args:
            diretorio_entrada: Diretório contendo os PDFs
            caminho_saida: Caminho para salvar embeddings
        """
        self._imprimir_cabecalho("📄 PROCESSANDO FAQs")
        
        arquivos_pdf = sorted(Path(diretorio_entrada).glob('*.pdf'))
        
        if not arquivos_pdf:
            print("⚠️  Nenhum arquivo PDF encontrado!")
            return
        
        perguntas_todas = []
        respostas_todas = []
        fontes_todas = []
        
        for arquivo_pdf in arquivos_pdf:
            print(f"📖 Processando: {arquivo_pdf.name}")
            
            texto = self.extrair_texto_pdf(str(arquivo_pdf))
            if not texto:
                continue
            
            perguntas, respostas = self.extrair_perguntas_respostas_de_texto(texto)
            
            # Adicionar à lista geral
            perguntas_todas.extend(perguntas)
            respostas_todas.extend(respostas)
            fontes_todas.extend([arquivo_pdf.stem] * len(perguntas))
        
        print(f"\n📊 Total de perguntas extraídas: {len(perguntas_todas)}")
        
        if not perguntas_todas:
            print("⚠️  Nenhuma pergunta foi extraída!")
            return
        
        # Gerar embeddings das respostas
        embeddings = self.gerar_embeddings(respostas_todas)
        
        # Preparar dados para salvar
        dados = {
            'tipo': 'faq',
            'total': len(perguntas_todas),
            'perguntas': perguntas_todas,
            'respostas': respostas_todas,
            'fontes': fontes_todas,
            'embeddings': embeddings.tolist()
        }
        
        self.salvar_embeddings(dados, caminho_saida)
    
    def processar_filmes(self, diretorio_entrada: str, caminho_saida: str) -> None:
        """
        Processa descrições de filmes e gera embeddings.
        
        Args:
            diretorio_entrada: Diretório contendo os arquivos TXT
            caminho_saida: Caminho para salvar embeddings
        """
        self._imprimir_cabecalho("🎬 PROCESSANDO FILMES")
        
        arquivos_txt = sorted(Path(diretorio_entrada).glob('*.txt'))
        
        if not arquivos_txt:
            print("⚠️  Nenhum arquivo TXT encontrado!")
            return
        
        titulos = []
        descricoes = []
        
        for arquivo_txt in arquivos_txt:
            print(f"📖 Processando: {arquivo_txt.name}")
            
            texto = self.extrair_texto_txt(str(arquivo_txt))
            if not texto:
                continue
            
            # Primeira linha é o título
            linhas = texto.split('\n', 1)
            titulo = linhas[0].strip()
            descricao = linhas[1].strip() if len(linhas) > 1 else texto
            
            titulos.append(titulo)
            descricoes.append(descricao)
        
        print(f"\n📊 Total de filmes: {len(titulos)}")
        
        if not descricoes:
            print("⚠️  Nenhuma descrição foi extraída!")
            return
        
        # Gerar embeddings das descrições
        embeddings = self.gerar_embeddings(descricoes)
        
        # Preparar dados para salvar
        dados = {
            'tipo': 'filmes',
            'total': len(titulos),
            'titulos': titulos,
            'descricoes': descricoes,
            'embeddings': embeddings.tolist()
        }
        
        self.salvar_embeddings(dados, caminho_saida)
    
    def processar_avaliacoes(self, arquivo_json: str, caminho_saida: str) -> None:
        """
        Processa feedbacks/avaliações e gera embeddings.
        
        Args:
            arquivo_json: Caminho para o arquivo JSON com feedbacks
            caminho_saida: Caminho para salvar embeddings
        """
        self._imprimir_cabecalho("⭐ PROCESSANDO AVALIAÇÕES")
        
        feedbacks = self.carregar_feedbacks_json(arquivo_json)
        
        if not feedbacks:
            print("⚠️  Nenhum feedback encontrado!")
            return
        
        print(f"📖 Carregados {len(feedbacks)} feedbacks")
        
        # Extrair dados dos feedbacks
        textos = [feedback['texto'] for feedback in feedbacks]
        ids = [feedback['id'] for feedback in feedbacks]
        usuarios = [feedback['usuario'] for feedback in feedbacks]
        datas = [feedback['data'] for feedback in feedbacks]
        
        # Gerar embeddings
        embeddings = self.gerar_embeddings(textos)
        
        # Preparar dados para salvar
        dados = {
            'tipo': 'avaliacoes',
            'total': len(feedbacks),
            'ids': ids,
            'usuarios': usuarios,
            'datas': datas,
            'textos': textos,
            'embeddings': embeddings.tolist()
        }
        
        self.salvar_embeddings(dados, caminho_saida)
    
    @staticmethod
    def _imprimir_cabecalho(titulo: str) -> None:
        """Imprime um cabeçalho formatado."""
        print("=" * 60)
        print(titulo)
        print("=" * 60)


def main():
    """Função principal que executa todo o processo de conversão."""
    print("\n" + "=" * 60)
    print("🚀 INICIANDO CONVERSÃO DE EMBEDDINGS")
    print("=" * 60 + "\n")
    
    # Definir caminhos base
    diretorio_base = Path(__file__).parent.parent
    diretorio_dados = diretorio_base / 'dados'
    diretorio_embeddings = diretorio_base / 'embeddings'
    
    # Inicializar conversor
    conversor = ConversorEmbeddings()
    
    # Processar FAQs
    conversor.processar_faqs(
        diretorio_entrada=str(diretorio_dados / 'faq'),
        caminho_saida=str(diretorio_embeddings / 'faq' / 'faq_embeddings.json')
    )
    
    # Processar Filmes
    conversor.processar_filmes(
        diretorio_entrada=str(diretorio_dados / 'filmes'),
        caminho_saida=str(diretorio_embeddings / 'filmes' / 'filmes_embeddings.json')
    )
    
    # Processar Avaliações
    conversor.processar_avaliacoes(
        arquivo_json=str(diretorio_dados / 'avaliacoes' / 'feedbacks.json'),
        caminho_saida=str(diretorio_embeddings / 'avaliacoes' / 'feedbacks_embeddings.json')
    )
    
    print("=" * 60)
    print("✅ CONVERSÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()