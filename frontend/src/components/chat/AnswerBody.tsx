import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

// The text of an answer: Markdown with GitHub tables, lists and links
export function AnswerBody({ text }: { text: string }) {
  return (
    <div className="prose prose-daisy prose-sm sm:prose-base prose-p:my-2 prose-p:whitespace-pre-wrap prose-li:my-0.5 prose-headings:my-2 max-w-none">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) =>
            href ? (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                {children}
              </a>
            ) : (
              <>{children}</>
            ),
          pre: ({ children }) => (
            <pre className="max-w-full overflow-x-auto">{children}</pre>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}
