#!/usr/bin/env python3
"""
busca_similaridade.py

Interface de linha de comando para buscar similaridade usando embeddings pré-gerados.
Oferece três modos de busca: FAQ, Filmes e Detecção de Anomalias.

Utiliza similaridade de cosseno para melhor separação entre scores.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss


class BuscadorSimilaridade:
    """Classe responsável por realizar buscas de similaridade usando FAISS."""
    
    MODELO_PADRAO = 'all-MiniLM-L6-v2'
    THRESHOLD_PADRAO = 0.6
    
    def __init__(self, modelo: str = MODELO_PADRAO):
        """
        Inicializa o buscador com o modelo especificado.
        
        Args:
            modelo: Nome do modelo Sentence Transformers
        """
        print(f"🔄 Carregando modelo: {modelo}")
        self.modelo = SentenceTransformer(modelo)
        print("✅ Modelo carregado!\n")
        self.threshold = self.THRESHOLD_PADRAO
    
    def carregar_embeddings(self, caminho: str) -> Optional[Dict]:
        """
        Carrega embeddings de um arquivo JSON.
        
        Args:
            caminho: Caminho para o arquivo de embeddings
            
        Returns:
            Dicionário com os dados e embeddings ou None se houver erro
        """
        try:
            with open(caminho, 'r', encoding='utf-8') as arquivo:
                dados = json.load(arquivo)
            
            # Converter embeddings de lista para numpy array
            dados['embeddings'] = np.array(dados['embeddings'], dtype=np.float32)
            return dados
        except Exception as erro:
            print(f"❌ Erro ao carregar embeddings: {erro}")
            return None
    
    @staticmethod
    def normalizar_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normaliza embeddings para ter norma L2 = 1.
        Necessário para usar similaridade de cosseno com produto interno.
        
        Args:
            embeddings: Array de embeddings
            
        Returns:
            Array de embeddings normalizados
        """
        normas = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / normas
    
    def criar_indice_faiss(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Cria um índice FAISS para busca por similaridade de cosseno.
        
        Args:
            embeddings: Array numpy com os embeddings
            
        Returns:
            Índice FAISS construído
        """
        dimensao = int(embeddings.shape[1])
        
        # Usar IndexFlatIP (Inner Product) para similaridade de cosseno
        indice = faiss.IndexFlatIP(dimensao)
        
        # Normalizar embeddings antes de adicionar
        embeddings_normalizados = self.normalizar_embeddings(embeddings)
        embeddings_float32 = embeddings_normalizados.astype(np.float32)
        
        indice.add(embeddings_float32)  # type: ignore
        
        return indice
    
    def buscar_similares(self, 
                        query: str, 
                        indice: faiss.IndexFlatIP, 
                        k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Busca os k itens mais similares à query usando similaridade de cosseno.
        
        Args:
            query: Texto da consulta
            indice: Índice FAISS
            k: Número de resultados a retornar
            
        Returns:
            Tupla com (scores_similaridade, índices) dos k mais similares
        """
        # Gerar embedding da query
        query_embedding = self.modelo.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Normalizar query
        query_normalizado = self.normalizar_embeddings(query_embedding)
        
        # Buscar no índice FAISS
        # Com IndexFlatIP e vetores normalizados, o resultado é a similaridade de cosseno
        similaridades, indices = indice.search(query_normalizado, k)  # type: ignore
        
        return similaridades[0], indices[0]
    
    @staticmethod
    def calcular_similaridade_cosseno(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula similaridade de cosseno entre dois embeddings.
        
        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding
            
        Returns:
            Similaridade de cosseno (0 a 1, onde 1 = idêntico)
        """
        # Normalizar
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Produto interno de vetores normalizados = similaridade de cosseno
        similaridade = float(np.dot(emb1_norm, emb2_norm))
        
        return similaridade
    
    def modo_faq(self, dados_embeddings: Dict) -> None:
        """
        Modo de busca para FAQs.
        
        Args:
            dados_embeddings: Dados carregados do arquivo de embeddings
        """
        self._imprimir_cabecalho_modo("📄 MODO FAQ - BUSCA SEMÂNTICA")
        
        print("Digite sua pergunta e encontraremos as respostas mais relevantes")
        print("das FAQs de: Advocacia, Engenharia Civil, Meio Ambiente,")
        print("Farmácia e Laboratório de Exames.\n")
        
        # Criar índice FAISS
        embeddings = dados_embeddings['embeddings']
        indice = self.criar_indice_faiss(embeddings)
        
        # Input do usuário
        query = input("❓ Digite sua pergunta: ").strip()
        
        if not query:
            print("⚠️  Query vazia. Encerrando modo FAQ.")
            return
        
        print(f"\n🔍 Buscando respostas para: '{query}'\n")
        
        # Buscar top 3
        scores, indices = self.buscar_similares(query, indice, k=3)
        
        # Exibir resultados
        print("=" * 60)
        print("🏆 TOP 3 RESPOSTAS MAIS RELEVANTES")
        print("=" * 60 + "\n")
        
        for i, (score, idx) in enumerate(zip(scores, indices), 1):
            pergunta = dados_embeddings['perguntas'][idx]
            resposta = dados_embeddings['respostas'][idx]
            fonte = dados_embeddings['fontes'][idx]
            
            print(f"#{i} - Fonte: {fonte}")
            print(f"Pergunta: {pergunta}")
            print(f"Resposta: {resposta[:200]}{'...' if len(resposta) > 200 else ''}")
            print("-" * 60 + "\n")
    
    def modo_filmes(self, dados_embeddings: Dict) -> None:
        """
        Modo de busca para recomendação de filmes.
        
        Args:
            dados_embeddings: Dados carregados do arquivo de embeddings
        """
        self._imprimir_cabecalho_modo("🎬 MODO FILMES - RECOMENDAÇÃO POR DESCRIÇÃO")
        
        print("Descreva o tipo de filme que você procura")
        print("(ex: 'viagem pelo tempo e espaço', 'sobrevivência em ambiente hostil')\n")
        
        # Criar índice FAISS
        embeddings = dados_embeddings['embeddings']
        indice = self.criar_indice_faiss(embeddings)
        
        # Input do usuário
        query = input("🎥 Descreva o filme que procura: ").strip()
        
        if not query:
            print("⚠️  Descrição vazia. Encerrando modo filmes.")
            return
        
        print(f"\n🔍 Buscando filmes para: '{query}'\n")
        
        # Buscar TODOS os filmes
        total_filmes = len(dados_embeddings['titulos'])
        scores, indices = self.buscar_similares(query, indice, k=total_filmes)
        
        # Exibir resultados
        print("=" * 60)
        print("🎬 FILMES RANQUEADOS POR SIMILARIDADE")
        print("=" * 60 + "\n")
        
        for i, (score, idx) in enumerate(zip(scores, indices), 1):
            titulo = dados_embeddings['titulos'][idx]
            descricao = dados_embeddings['descricoes'][idx]
            
            print(f"#{i} - {titulo}")
            print(f"   {descricao[:150]}{'...' if len(descricao) > 150 else ''}")
            print()
    
    def modo_anomalias(self, dados_embeddings: Dict) -> None:
        """
        Modo de detecção de anomalias em feedbacks.
        Detecta feedbacks com baixa similaridade média com os demais.
        
        Args:
            dados_embeddings: Dados carregados do arquivo de embeddings
        """
        self._imprimir_cabecalho_modo("⚠️  MODO DETECÇÃO DE ANOMALIAS")
        
        print("Analisando feedbacks para identificar conteúdos anômalos...")
        print("(feedbacks que diferem significativamente dos demais)\n")
        
        embeddings = dados_embeddings['embeddings']
        total = len(embeddings)
        
        print(f"📊 Total de feedbacks: {total}\n")
        print("🔄 Calculando similaridade média de cada feedback...\n")
        
        # Para cada feedback, calcular similaridade média com todos os outros
        scores_medios = []
        
        for i in range(total):
            embedding_atual = embeddings[i]
            similaridades = []
            
            for j in range(total):
                if i != j:
                    embedding_outro = embeddings[j]
                    similaridade = self.calcular_similaridade_cosseno(
                        embedding_atual, 
                        embedding_outro
                    )
                    similaridades.append(similaridade)
            
            # Média de similaridade
            score_medio = float(np.mean(similaridades))
            scores_medios.append((i, score_medio))
        
        # Ordenar por score médio (menor = mais anômalo)
        scores_medios.sort(key=lambda x: x[1])
        
        # Exibir top 5 anomalias
        print("=" * 60)
        print("🚨 TOP 5 FEEDBACKS ANÔMALOS DETECTADOS")
        print("=" * 60 + "\n")
        
        for rank, (idx, score_medio) in enumerate(scores_medios[:3], 1):
            feedback_id = dados_embeddings['ids'][idx]
            usuario = dados_embeddings['usuarios'][idx]
            data = dados_embeddings['datas'][idx]
            texto = dados_embeddings['textos'][idx]
            
            print(f"🚨 Anomalia #{rank}")
            print(f"   ID: {feedback_id} | Usuário: {usuario} | Data: {data}")
            print(f"   Feedback: {texto}")
            print(f"   {'=' * 60}\n")
    
    @staticmethod
    def _imprimir_cabecalho_modo(titulo: str) -> None:
        """Imprime cabeçalho formatado para cada modo."""
        print("\n" + "=" * 60)
        print(titulo)
        print("=" * 60 + "\n")


def exibir_menu() -> str:
    """
    Exibe o menu principal e captura a escolha do usuário.
    
    Returns:
        Opção escolhida pelo usuário
    """
    print("\n" + "=" * 60)
    print("🔍 SISTEMA DE BUSCA POR SIMILARIDADE - EMBEDDINGS")
    print("=" * 60)
    print("\nEscolha o modo de busca:\n")
    print("  1️⃣  - Modo FAQ (Busca Semântica em Perguntas Frequentes)")
    print("  2️⃣  - Modo Filmes (Recomendação por Descrição)")
    print("  3️⃣  - Modo Anomalias (Detecção de Feedbacks Anômalos)")
    print("  0️⃣  - Sair\n")
    
    escolha = input("👉 Digite sua escolha [1/2/3/0]: ").strip()
    return escolha


def main():
    """Função principal com loop do menu."""
    
    # Definir caminhos
    diretorio_base = Path(__file__).parent.parent
    diretorio_embeddings = diretorio_base / 'embeddings'
    
    # Caminhos dos arquivos de embeddings
    caminho_faq = diretorio_embeddings / 'faq' / 'faq_embeddings.json'
    caminho_filmes = diretorio_embeddings / 'filmes' / 'filmes_embeddings.json'
    caminho_avaliacoes = diretorio_embeddings / 'avaliacoes' / 'feedbacks_embeddings.json'
    
    arquivos_necessarios = [caminho_faq, caminho_filmes, caminho_avaliacoes]
    arquivos_faltando = [arquivo for arquivo in arquivos_necessarios if not arquivo.exists()]
    
    if arquivos_faltando:
        print("\n❌ ERRO: Arquivos de embeddings não encontrados!")
        print("\nArquivos faltando:")
        for arquivo in arquivos_faltando:
            print(f"  - {arquivo}")
        print("\n⚠️  Execute primeiro: python src/converte_embeddings.py\n")
        return
    
    # Inicializar buscador
    buscador = BuscadorSimilaridade()
    
    # Loop do menu
    while True:
        escolha = exibir_menu()
        
        if escolha == '1':
            # Modo FAQ
            dados = buscador.carregar_embeddings(str(caminho_faq))
            if dados:
                buscador.modo_faq(dados)
        
        elif escolha == '2':
            # Modo Filmes
            dados = buscador.carregar_embeddings(str(caminho_filmes))
            if dados:
                buscador.modo_filmes(dados)
        
        elif escolha == '3':
            # Modo Anomalias
            dados = buscador.carregar_embeddings(str(caminho_avaliacoes))
            if dados:
                buscador.modo_anomalias(dados)
        
        elif escolha == '0':
            print("\n👋 Encerrando sistema. Até logo!\n")
            break
        
        else:
            print("\n⚠️  Opção inválida. Tente novamente.")
        
        input("\n⏸️  Pressione ENTER para voltar ao menu...")


if __name__ == '__main__':
    main()