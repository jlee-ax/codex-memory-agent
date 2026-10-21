# codex-memory-agent

Python·Jupyter에서 각자의 Codex 로그인으로 요청하고, 에이전트가 SQLite 외부 메모리를 조회·수정하며 분류 체계를 만드는 패키지입니다.

두 계층을 독립적으로 사용할 수 있습니다.

| 계층 | 인터페이스 | 역할 |
| --- | --- | --- |
| 연결 | `CodexConnection` | 로컬 Codex App Server와 JSON-RPC로 요청·응답 및 사용자 정의 도구 호출을 처리합니다. |
| 분류 | `TaxonomyAgent`, `TaxonomyMemory` | 원문을 나누어 읽고 범주를 생성·수정·병합·분할하며 전체 항목을 최종 재검토합니다. |

벡터화나 고정된 k를 사용하지 않습니다. 범주와 판단 근거는 SQLite에 저장되며, 대화 기록 대신 저장된 상태를 다음 요청에서 조회합니다. 모델 가중치를 학습하는 방식은 아닙니다.

## 시작하기

Python 3.12 이상, FTS5를 지원하는 SQLite, 설치된 Codex CLI가 필요합니다. 각 사용자는 자신의 계정으로 `codex login`을 수행합니다. Python 핵심 패키지는 외부 의존성이 없습니다.

```bash
python -m pip install 'git+https://github.com/jlee-ax/codex-memory-agent.git'
```

Jupyter 커널에는 다음과 같이 설치합니다.

```python
%pip install "codex-memory-agent[notebook] @ git+https://github.com/jlee-ax/codex-memory-agent.git"
```

```python
from codex_memory_agent import TaxonomyAgent

texts = ["Draft a proposal", "Debug a function", "Summarize feedback"]
prompt = "Group tasks by work performed. Start with no categories and let k emerge from evidence."

agent = TaxonomyAgent()
agent.start(texts, prompt, directory=".codex-memory-agent/my-project")
agent.monitor()
result = agent.results()
print(result["k"])
print(result["assignments"])
```

`agent.stop()` 또는 모니터링 중 Ctrl-C로 중단합니다. 같은 원문·프롬프트·저장 위치로 다시 `start()`하면 저장된 판단에서 재개합니다. 데이터나 프롬프트가 달라지면 별도 DB가 생성됩니다. 사용을 마치면 `agent.close()`를 실행합니다. Jupyter 커널은 종료하지 않습니다.

입력은 서로 다른 원문 문자열을 기준으로 분류합니다. 중복 문자열은 한 번만 처리하며, `raw_text`로 원래 테이블에 결합할 수 있습니다. 최종 결과는 `result["status"]["complete"]`로 확인합니다. 범주 수정으로 검토 기준이 달라지면 현재 버전을 기준으로 전체 재검토가 필요합니다.

이 방식은 맥락 재전송을 줄이지만, 모든 항목을 읽고 검토하는 모델 처리 비용까지 제거하지는 않습니다. 짧은 텍스트 레코드에 적합하며 긴 문서는 의미 있는 단위로 나누어 입력해야 합니다.

인증은 로컬 오픈소스 애플리케이션에서 각자의 계정을 사용하는 범위로 설계했습니다. 서비스 운영에 대한 인증 범위와 현재 제한은 [OpenAI 공식 문서](https://learn.chatgpt.com/docs/app-server)를 확인하세요.

[영문 API 안내](README.md) · [두 체계의 알고리즘](docs/architecture.md) · [예제](examples)
