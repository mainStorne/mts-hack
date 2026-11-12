// A neutral logo of the demo brand: a chat bubble
export function Logo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      className={className}
      aria-hidden="true"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="64" height="64" rx="16" className="fill-primary" />
      <path
        d="M18 18h28a6 6 0 0 1 6 6v14a6 6 0 0 1-6 6H33l-9 8v-8h-6a6 6 0 0 1-6-6V24a6 6 0 0 1 6-6Z"
        className="fill-primary-content"
      />
      <circle cx="24" cy="31" r="3" className="fill-primary" />
      <circle cx="32" cy="31" r="3" className="fill-primary" />
      <circle cx="40" cy="31" r="3" className="fill-primary" />
    </svg>
  );
}
