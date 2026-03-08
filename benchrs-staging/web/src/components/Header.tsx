import { FlaskConical, Github } from "lucide-react";

export default function Header() {
  return (
    <header className="border-b border-gray-100 bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex items-center gap-2.5">
          <FlaskConical className="h-5 w-5 text-amber-500" />
          <span className="text-lg font-bold tracking-tight text-gray-900">
            benchrs
          </span>
          <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-medium text-gray-500">
            v0.1.0
          </span>
        </div>
        <nav className="flex items-center gap-4 text-sm text-gray-500">
          <a href="#methodology" className="hover:text-gray-900">
            Methodology
          </a>
          <a
            href="https://github.com/goliajp/airs"
            className="flex items-center gap-1 hover:text-gray-900"
          >
            <Github className="h-4 w-4" />
            <span className="hidden sm:inline">Source</span>
          </a>
        </nav>
      </div>
    </header>
  );
}
