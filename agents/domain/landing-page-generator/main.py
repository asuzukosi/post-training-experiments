"""cli entry for landing page generator langgraph pipeline."""

from __future__ import annotations
import os
import shutil
from textwrap import dedent
from graph import build_landing_graph


def main() -> None:
    print("welcome to idea generator")
    print(
        dedent(
            """
  ! you must fork this before using it !
  """
        )
    )
    print(
        dedent(
            """
      disclaimer: this will use your configured model and may cost money (~2-9 usd).
      the full run might take around ~10-45m. enjoy your time back.

    """
        )
    )
    idea = input("# describe what is your idea:\n\n").strip()

    if not os.path.exists("./workdir"):
        os.mkdir("./workdir")

    if len(os.listdir("./templates")) == 0:
        print(
            dedent(
                """
      !!! no templates found !!!
      ! you must fork this before using it !

      templates are not included as they are tailwind templates.
      place tailwind individual template folders in `./templates`,
      if you have a license you can download them at
      https://tailwindui.com/templates, their references are at
      `config/templates.json`.

      this was not tested with other templates,
      prompts in `tasks.py` might require some changes
      for that to work.

      !!! stopping execution !!!
      """
            )
        )
        return

    graph = build_landing_graph()
    graph.invoke(
        {
            "messages": [{"role": "user", "content": idea}],
            "output": "",
            "idea": idea,
            "expanded_idea": "",
            "components": [],
            "component_index": 0,
        }
    )

    zip_file = "workdir"
    shutil.make_archive(zip_file, "zip", "workdir")
    shutil.rmtree("workdir")
    print("\n\n")
    print("==========================================")
    print("done!")
    print(f"you can download the project at ./{zip_file}.zip")
    print("==========================================")


if __name__ == "__main__":
    main()
